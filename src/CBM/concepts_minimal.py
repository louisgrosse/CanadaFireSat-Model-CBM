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
import yaml  # read OmegaConf-saved config.yaml

# Ensure local imports (your uploaded files) work
_THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(_THIS_DIR))

# Your SAE module (Lightning)
try:
    from sae import plSAE  # must expose plSAE with .net and .net.get_dictionary()
except Exception as e:
    raise ImportError(f"Could not import plSAE from sae.py: {e}")


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
    try:
        sys.path.append("MS-CLIP")
        from msclip.inference.utils import build_model as msclip_build  # type: ignore
        msclip_model, _, tokenizer = msclip_build(
            model_name=model_name,
            pretrained=True,
            ckpt_path=ckpt_path,
            device=device,
            channels=10
        )

        def _encode_text(texts: List[str]) -> torch.Tensor:
            toks = tokenizer(texts)
            if isinstance(toks, dict) and "input_ids" in toks:  # HF-style token dict
                input_ids = torch.as_tensor(toks["input_ids"]).to(device)
                return msclip_model.text_encoder.encode_text(input_ids)  # type: ignore
            else:
                toks = torch.as_tensor(toks).to(device)
                return msclip_model.text_encoder.encode_text(toks)  # type: ignore

        return _encode_text, tokenizer
    except Exception:
        pass

    # Fallback: OpenCLIP
    try:
        import open_clip  # type: ignore
        oc_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b-s34b-b88K",
            device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        oc_model.eval()

        @torch.no_grad()
        def _encode_text(texts: List[str]) -> torch.Tensor:
            toks = tokenizer(texts)
            toks = torch.as_tensor(toks).to(device)
            feats = oc_model.encode_text(toks)
            return feats

        return _encode_text, tokenizer
    except Exception as e:
        raise ImportError(f"Could not set up any text encoder (MS-CLIP or OpenCLIP): {e}")


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


# ---------------------------
# SAE config loading (exact schema)
# ---------------------------

def _read_cfg_yaml(ckpt_path: str) -> Dict[str, Any]:
    """
    Load OmegaConf-saved YAML expected at <ckpt_dir>/config.yaml.
    """
    ckpt_dir = Path(ckpt_path).resolve().parent
    yml = ckpt_dir / "config.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"config.yaml not found next to checkpoint: {yml}")
    with open(yml, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _resolve_interpolations_exact(sae_block: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve the specific interpolations you showed:

      - sae.sae_kwargs.input_shape: "${input_shape}" -> cfg["input_shape"]
      - sae.sae_kwargs.nb_concepts: "${nb_concepts}" -> cfg["nb_concepts"]
      - sae.sae_kwargs.top_k: "${nb_k}" -> cfg["nb_k"]
      - sae.sae_kwargs.device: '${oc.select:device, "cpu"}' -> "cpu"

    No other guessing or schema probing is performed.
    """
    if not isinstance(sae_block, dict):
        raise ValueError("cfg['sae'] must be a dict.")

    sae_block = json.loads(json.dumps(sae_block))  # deep copy (avoid in-place on caller)
    sae_kwargs = sae_block.get("sae_kwargs", {})
    if not isinstance(sae_kwargs, dict):
        sae_kwargs = {}
        sae_block["sae_kwargs"] = sae_kwargs

    # 1) input_shape
    v = sae_kwargs.get("input_shape", None)
    if isinstance(v, str) and v.strip() == "${input_shape}":
        if "input_shape" not in cfg:
            raise KeyError("config.yaml missing top-level 'input_shape' required by sae.sae_kwargs.input_shape")
        sae_kwargs["input_shape"] = cfg["input_shape"]

    # 2) nb_concepts
    v = sae_kwargs.get("nb_concepts", None)
    if isinstance(v, str) and v.strip() == "${nb_concepts}":
        if "nb_concepts" not in cfg:
            raise KeyError("config.yaml missing top-level 'nb_concepts' required by sae.sae_kwargs.nb_concepts")
        sae_kwargs["nb_concepts"] = cfg["nb_concepts"]

    # 3) top_k
    v = sae_kwargs.get("top_k", None)
    if isinstance(v, str) and v.strip() == "${nb_k}":
        if "nb_k" not in cfg:
            raise KeyError("config.yaml missing top-level 'nb_k' required by sae.sae_kwargs.top_k")
        sae_kwargs["top_k"] = cfg["nb_k"]

    # 4) device
    v = sae_kwargs.get("device", None)
    if isinstance(v, str) and v.strip().startswith("${oc.select:device"):
        # You explicitly showed '${oc.select:device, "cpu"}' -> we set "cpu"
        sae_kwargs["device"] = "cpu"

    sae_block["sae_kwargs"] = sae_kwargs
    return sae_block


def _load_sae_from_ckpt(ckpt_path: str, device: str = "cpu") -> plSAE:
    """
    Strict, schema-accurate loader:
      - cfg = YAML at <ckpt_dir>/config.yaml
      - sae_block = cfg["sae"] (must exist)
      - resolve only the specific interpolations listed above
      - try Lightning load_from_checkpoint(**sae_block)
      - fallback: instantiate plSAE(**sae_block) and load state dict
    """
    cfg = _read_cfg_yaml(ckpt_path)
    if "sae" not in cfg or not isinstance(cfg["sae"], dict):
        raise KeyError("config.yaml must contain a 'sae' block matching your schema.")

    sae_block = _resolve_interpolations_exact(cfg["sae"], cfg)

    # 1) Try Lightning load with the config block as-is
    try:
        sae = plSAE.load_from_checkpoint(ckpt_path, map_location=device, **sae_block)
        sae.eval().to(device)
        return sae
    except Exception:
        pass

    # 2) Fallback: manual construct from the same sae_block, then load state_dict
    sae = plSAE(**sae_block)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        sae.load_state_dict(state["state_dict"], strict=False)
    else:
        sae.load_state_dict(state, strict=False)
    sae.eval().to(device)
    return sae


# ---------------------------
# Concept naming via cosine
# ---------------------------

@torch.no_grad()
def name_sae_concepts(sae_ckpt: str,
                      csv_path: str,
                      text_col: str = "concept",
                      topk: int = 5,
                      model_name: str = "Llama3-MS-CLIP-Base",
                      msclip_ckpt: Optional[str] = None,
                      device: str = "cpu",
                      out_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Match SAE dictionary vectors to text embeddings by cosine similarity.
    Returns DataFrame with: concept_id, top1_name, top1_sim, topk_names, topk_sims
    """
    sae = _load_sae_from_ckpt(sae_ckpt, device)
    Dmat = sae.net.get_dictionary()  # expect [H, D] (columns are features)
    if isinstance(Dmat, np.ndarray):
        Dmat = torch.from_numpy(Dmat)
    Dmat = Dmat.float().to(device)

    # Coerce to [H, D] if needed
    if Dmat.dim() == 2 and Dmat.shape[0] > Dmat.shape[1]:
        Dmat = Dmat.t().contiguous()
    Dmat = F.normalize(Dmat, dim=-1)

    txt = csv_text_embeddings(
        csv_path, text_col=text_col, model_name=model_name, ckpt_path=msclip_ckpt, device=device
    )
    phrases: List[str] = txt["phrases"]
    E = txt["embs"].to(device)  # [N, D]

    # Cosine similarities: [H, D] @ [D, N] = [H, N]
    sims = Dmat @ E.t()
    topk_vals, topk_idx = torch.topk(sims, k=min(topk, E.shape[0]), dim=1)

    rows = []
    for c in range(Dmat.shape[0]):
        names = [phrases[j] for j in topk_idx[c].cpu().tolist()]
        vals = topk_vals[c].cpu().tolist()
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


# ---------------------------
# Optional: concept bottleneck (frozen SAE + 1×1 head)
# ---------------------------

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
