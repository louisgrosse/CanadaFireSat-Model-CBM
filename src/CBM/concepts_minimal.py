import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml  
from omegaconf import OmegaConf

sys.path.append("MS-CLIP")
from msclip.inference.utils import build_model as msclip_build  


from src.models.sae import plSAE

_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR))

# ---------------------------
# MS-CLIP / OpenCLIP text side
# ---------------------------

def _load_msclip_text_encoder(model_name: str = "Llama3-MS-CLIP-Base",
                              ckpt_path: Optional[str] = None,
                              device: str = "cpu"):
    """
    Prefer the MS-CLIP builder exactly as used in your codebase.
    If that import fails, fallback to OpenCLIP ViT-B/16.
    """
    
    msclip_model, _, tokenizer = msclip_build(
        model_name=model_name,
        pretrained=True,
        ckpt_path=ckpt_path,
        device=device,
        channels=10
    )

    msclip_model = msclip_model.to("cuda").eval()

    def _encode_text(texts: List[str]) -> torch.Tensor:
        toks = tokenizer(texts)
        model_device = next(msclip_model.parameters()).device

        toks = torch.as_tensor(toks, dtype=torch.long, device=model_device)
        return msclip_model.clip_base_model.model.encode_text(toks)  

    return _encode_text, tokenizer



@torch.no_grad()
def csv_text_embeddings(csv_path: str,
                        text_col: str = "concept",
                        model_name: str = "Llama3-MS-CLIP-Base",
                        ckpt_path: Optional[str] = None,
                        device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Embed CSV[text_col] phrases via MS-CLIP text encoder (or OpenCLIP fallback).
    Returns {"phrases": List[str], "embs": torch.tensor [N,D] L2-normalized}.
    """
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"CSV lacks a '{text_col}' column. Columns found: {list(df.columns)}")

    phrases_raw = df[text_col].astype(str).fillna("").tolist()
    phrases = [p.strip() for p in phrases_raw if p.strip()]
    if not phrases:
        raise ValueError("No non-empty phrases to embed.")

    encode_text, _ = _load_msclip_text_encoder(
        model_name=model_name, ckpt_path=ckpt_path, device=device
    )

    batch_size = 256
    out = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        feats = encode_text(batch).float().to(device)
        feats = F.normalize(feats, dim=-1)
        out.append(feats)
    embs = torch.cat(out, dim=0)  # [N,D]
    return {"phrases": phrases, "embs": embs}


def _load_sae_from_ckpt(ckpt_path: str, device: str = "cpu"):
    cfg = OmegaConf.load(os.path.join(Path(ckpt_path).parent, "config.yaml"))
    sae_kwargs = OmegaConf.to_container(cfg["sae"], resolve=True)
    sae_kwargs.pop("_target_", None) 
    sae = plSAE.load_from_checkpoint(ckpt_path, map_location=device, **sae_kwargs)
    return sae.eval().to(device)

@torch.no_grad()
def name_sae_concepts(sae_ckpt: str,
                      csv_path: str,
                      text_col: str = "concept",
                      topk: int = 5,
                      model_name: str = "Llama3-MS-CLIP-Base",
                      msclip_ckpt: Optional[str] = None,
                      device: str = "cpu",
                      out_csv: Optional[str] = None) -> pd.DataFrame:
    sae = _load_sae_from_ckpt(sae_ckpt, device)
    Dmat = sae.net.get_dictionary()
    if isinstance(Dmat, np.ndarray):
        Dmat = torch.from_numpy(Dmat)
    Dmat = Dmat.float().to(device)  # [?, ?]

    # Ensure [H, D] (rows=concepts, cols=feat-dim)
    if Dmat.dim() == 2 and Dmat.shape[1] > Dmat.shape[0]:
        # looks like [D, H] -> transpose
        Dmat = Dmat.t().contiguous()  # [H, D]

    # Text embeddings
    txt = csv_text_embeddings(csv_path, text_col=text_col,
                              model_name=model_name, ckpt_path=msclip_ckpt, device=device)
    phrases: List[str] = txt["phrases"]
    E = txt["embs"].to(device)  # [N, D_text]

    # If dictionary dim != text dim, project dictionary to CLIP text space
    if Dmat.shape[1] != E.shape[1]:
        msclip_model, _, _ = msclip_build(model_name=model_name, pretrained=True,
                                          ckpt_path=msclip_ckpt, device=device, channels=10)
        msclip_model = msclip_model.to(device).eval()
        # OpenCLIP visual projection: [vit_width, text_dim] e.g. [768, 512]
        W = msclip_model.clip_base_model.model.visual.proj.to(device).float()
        # Dmat is [H, vit_width] -> [H, text_dim]
        if Dmat.shape[1] == W.shape[0]:
            Dmat = Dmat @ W
        else:
            raise RuntimeError(
                f"Cannot align dims: Dmat {tuple(Dmat.shape)} vs proj {tuple(W.shape)} vs text {tuple(E.shape)}"
            )

    # Final shape check: [H, D_text]
    if Dmat.shape[1] != E.shape[1]:
        raise RuntimeError(f"Post-proj mismatch: Dmat {tuple(Dmat.shape)} vs text {tuple(E.shape)}")

    # Normalize once, then cosine
    Dmat = F.normalize(Dmat, dim=-1)
    E = F.normalize(E, dim=-1)

    sims = Dmat @ E.t()  # [H, N]
    k = min(topk, E.shape[0]) if E.shape[0] > 0 else 1
    topk_vals, topk_idx = torch.topk(sims, k=k, dim=1)

    rows = []
    for c in range(Dmat.shape[0]):
        idxs = topk_idx[c].tolist()
        names = [phrases[j] for j in idxs]
        vals = topk_vals[c].tolist()
        rows.append({
            "concept_id": c,
            "top1_name": names[0],
            "top1_sim": float(vals[0]),
            "topk_names": json.dumps(names),
            "topk_sims": json.dumps([float(v) for v in vals]),
        })
    df = pd.DataFrame(rows)

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df


class SAEConceptBottleneck(nn.Module):
    """
    Frozen SAE -> codes per patch -> 1x1 conv head (C_out channels).
    Inputs:
      x_feats: [B, D, H, W] pre-head patch embeddings (same D SAE was trained on).
    """
    def __init__(self, sae_ckpt: str, C_out: int = 2, device: str = "cpu", trainable_head: bool = True):
        super().__init__()
        self.device_str = device
        self.sae = _load_sae_from_ckpt(sae_ckpt, device=device)
        for p in self.sae.parameters():
            p.requires_grad = False
        H_codes = self.sae.net.get_dictionary().shape[0]
        self.head = nn.Conv2d(H_codes, C_out, kernel_size=1, bias=True)
        self.head.weight.requires_grad = trainable_head
        self.head.bias.requires_grad = trainable_head

    def forward(self, x_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = x_feats.shape
        x = x_feats.permute(0, 2, 3, 1).contiguous().view(B*H*W, D).to(self.sae.device)
        x = x.float()
        # Standardize feature dims like typical SAE training (adjust if your SAE expects different)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, unbiased=False, keepdim=True) + 1e-6)
        z_pre, z, x_hat = self.sae.net(x)  # z: [B*H*W, H_codes]
        codes = z.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, H_codes, H, W]
        logits = self.head(codes)  # [B, C_out, H, W]
        return logits, codes


# ---------------------------
# Quick viz of top concept per patch
# ---------------------------

@torch.no_grad()
def visualize_top_concept_per_patch(codes: torch.Tensor,
                                    names_df: pd.DataFrame,
                                    out_path: Optional[str] = None):
    """
    Render a heatmap of the top-activating concept id per patch for B=1.
    """
    import matplotlib.pyplot as plt

    B, Hc, H, W = codes.shape
    assert B == 1, "For simplicity, pass a single example (B=1)."
    top_idx = codes[0].argmax(dim=0).cpu().numpy()  # [H,W]

    concept_names = names_df.set_index("concept_id")["top1_name"].to_dict()
    label_img = np.empty((H, W), dtype=object)
    for i in range(H):
        for j in range(W):
            cid = int(top_idx[i, j])
            name = concept_names.get(cid, str(cid))
            if len(name) > 10:
                name = name[:9] + "…"
            label_img[i, j] = name

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(top_idx, interpolation='nearest')
    ax.set_title("Top concept id per patch")
    ax.set_xticks([]); ax.set_yticks([])
    # annotate sparsely to keep readable
    for i in range(H):
        for j in range(W):
            if (i % 2 == 0) and (j % 2 == 0):
                ax.text(j, i, str(label_img[i, j]), ha="center", va="center", fontsize=6)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    return fig
