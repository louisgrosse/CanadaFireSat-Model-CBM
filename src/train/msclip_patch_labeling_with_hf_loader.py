
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import sys
from src.data import get_data
from src.models import get_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.torch_utils import load_from_checkpoint


from src.constants import CONFIG_PATH
from src.data.Canada.data_transforms import Canada_segmentation_transform, MSCLIP_MEANS, MSCLIP_STDS

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from msclip.inference.clearclip import maybe_patch_clearclip
from msclip.inference.sclip import maybe_patch_sclip

@torch.no_grad()
def prehead_forward_sae(model: torch.nn.Module, x: torch.Tensor, doy: Optional[torch.Tensor] = None) -> torch.Tensor:
    #[B, T, C, H, W] 
    B, T, C, H, W = x.shape

    x = x.reshape(B * T, C, H, W)

    requires_enc_grad = any(p.requires_grad for p in model.image_encoder.parameters())
    ctx = torch.enable_grad() if requires_enc_grad and model.training else torch.no_grad()
    with ctx:
        pooled_feats, patch_feats = model.msclip_model.image_encoder(x)

    # [B, T, P, D]
    patch_feats = patch_feats.view(B, T, model.num_patches, model.embed_dim)

    if model.use_l1c2l2a_adapter:
        patch_feats = model.l1c2l2a(patch_feats.view(B*T, model.num_patches, model.embed_dim)).view(
            B, T, model.num_patches, model.embed_dim
        )

    # (B*P, T, D)
    patch_feats = patch_feats.permute(0, 2, 1, 3).contiguous().view(B * model.num_patches, T, model.embed_dim)

    doy_emb = None
    if model.use_doy and doy is not None:
        assert doy.shape[0] == B and doy.shape[1] == T, f"DOY shape mismatch: {doy.shape} vs {(B,T)}"
        if doy.ndim > 2:  # [B,T,H,W,1] -> [B,T]
            doy = doy.view(B, T, -1)[:, :, 0]
        assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
        assert torch.isfinite(doy).all(), "DOY has NaN/Inf"

        doy = doy.clamp(0, 1)
        doy_emb = model.doy_embed(doy)  # [B,T,D]
        doy_emb = doy_emb.unsqueeze(1).expand(-1, model.num_patches, -1, -1).reshape(
            B * model.num_patches, T, model.embed_dim
        )
    else:
        doy_emb = None


    patch_feats = model.temp_enc(patch_feats, doy_emb=doy_emb)  # [B*P, D]
    assert torch.isfinite(patch_feats).all(), "Temporal encoder produced NaN/Inf"

    # [B, P, D]
    patch_feats = patch_feats.view(B, model.num_patches, model.embed_dim)
    if model.use_cls_fusion and model.has_cls_token:
        cls_feats = cls_feats.view(B, T, model.embed_dim)
        cls_feats = model.cls_temp_enc(cls_feats)
        patch_feats = patch_feats + model.cls_fuse_proj(cls_feats).unsqueeze(1)

    model.sae(patch_feats)
    # [B,D,H_p,W_p]
    patch_feats = patch_feats.view(B, model.H_patch, model.W_patch, model.embed_dim).permute(0, 3, 1, 2).contiguous()

    return patch_feats

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_captions_dictionary(captions_dir: str, pattern: str = "*.parquet",
                             max_phrases: Optional[int] = 50000) -> List[str]:
    path = sorted(glob.glob(os.path.join(captions_dir, pattern)))

    phrases: List[str] = []

    df = pd.read_parquet(path)
    print(f"[WARN] Failed to read {p}: {e}")
    taken = False
    for col in ["caption", "captions", "text", "description", "prompt"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            phrases.extend(vals); taken = True
    if not taken:
        
        for col in df.columns:
            if df[col].dtype == object:
                phrases.extend(df[col].dropna().astype(str).tolist())

    seen = set(); uniq = []
    for s in phrases:
        s = s.strip()
        if not s or s in seen: continue
        seen.add(s); uniq.append(s)
    if max_phrases is not None:
        if len(uniq) > max_phrases:
            uniq = uniq[:max_phrases]
    print(f"[INFO] Loaded {len(uniq)} unique phrases from {len(paths)} parquet file(s).")
    return uniq

def _binarize_labels(y, x=None):
    """
    Normalize label tensor shape to [B,1,H,W] of 0/1 floats.
    Returns None if we can't interpret y.
    """
    if y is None:
        return None

    # y can be [B,H,W]
    if y.ndim == 3:
        yb = (y.unsqueeze(1) > 0.5).float()
        return yb

    # y can be [B,1,H,W] already
    if y.ndim == 4:
        B, C, H, W = y.shape
        # If it's single-channel logits or probs
        if C == 1:
            return (y > 0.5).float()

        # Could be one-hot or logits for K classes
        # We'll take argmax == 1 (class "fire") as positive mask, same as your OcclusionMaps.py
        # Adjust if your positive class is a different index.
        with torch.no_grad():
            if C > 1:
                cls_idx = 1  # fire class
                if y.dtype.is_floating_point:
                    probs = torch.softmax(y, dim=1)
                    fire_mask = (probs.argmax(dim=1, keepdim=True) == cls_idx).float()
                else:
                    fire_mask = (y.argmax(dim=1, keepdim=True) == cls_idx).float()
                return fire_mask
    # otherwise we don't know how to interpret
    return None

def _get_positive_indices(batch, min_pos_frac=0.5):
    """
    Returns list of indices within this batch that are 'positive'.
    Positive == label mask >0.5 occupies >= min_pos_frac of pixels.
    """
    x = batch["inputs"]  # [B,T,C,H,W]
    y = batch["labels"]
    yb = _binarize_labels(y, x)

    if yb is None:
        return []

    B = yb.shape[0]
    pos_ratio = (yb > 0.5).float().view(B, -1).mean(dim=1)
    keep = (pos_ratio >= min_pos_frac).nonzero(as_tuple=True)[0]
    return keep.tolist()

def tokenize_phrases(tokenizer, phrases: List[str], batch_size: int, device: str, model) -> torch.Tensor:
    
    embs = []
    for i in tqdm(range(0, len(phrases), batch_size)):
        batch = phrases[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        z = model.inference_text(tokens)
        z = F.normalize(z, dim=-1)
        embs.append(z)
    return torch.cat(embs, dim=0)

def assign_labels_to_patches(patch_embs: torch.Tensor, text_embs: torch.Tensor, topk: int = 1):
    """
    patch_embs: [B,P,D] or [B,D] or [P,D]
    text_embs:  [N,D]
    returns (scores, idx):
      scores: [B,P,topk]  cosine sim of top phrases
      idx:    [B,P,topk]  indices into text_embs / phrases
    """
    patch_embs = F.normalize(patch_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)
    if patch_embs.ndim == 2:
        # treat as [B,D] -> make P=1
        patch_embs = patch_embs.unsqueeze(1)  # [B,1,D]

    B, P, D = patch_embs.shape
    N = text_embs.shape[0]

    # [B,P,D] -> [B*P,D]
    flat = patch_embs.reshape(B * P, D)  # [BP,D]

    # [BP,N] similarities
    sims_flat = flat @ text_embs.t()     # both already L2-normalized so this is cosine

    # back to [B,P,N]
    sims = sims_flat.reshape(B, P, N)

    # topk over phrases
    scores, idx = torch.topk(sims, k=topk, dim=-1)  # each [B,P,topk]

    return scores, idx


def _wrap_phrase(s: str, max_len_line: int = 40, max_lines: int = 4) -> str:
    """
    Turn a long caption into multiple short lines so it fits in the legend panel
    without blasting across the figure. We also cap total lines.
    """
    s = s.replace("\n", " ").strip()
    words = s.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + 1 + len(w) > max_len_line:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
            if len(lines) >= max_lines:
                break
        else:
            cur.append(w)
            cur_len += (len(w) + (1 if cur_len > 0 else 0))
    if len(lines) < max_lines and cur:
        lines.append(" ".join(cur))
    # add ellipsis if we dropped content
    text = "\n".join(lines)
    if len(lines) >= max_lines and (len(words) > sum(len(l.split()) for l in lines)):
        text += " …"
    return text


def msclip_denorm_to_rgb(x_norm: torch.Tensor, rgb_bands=(3,2,1)) -> np.ndarray:
    """x_norm: [C,H,W] normalized with MSCLIP_MEANS/STDS on reflectance-like scale (after 10000/255 factor).
       Return quicklook RGB in [0,1].
    """
    means = torch.tensor(MSCLIP_MEANS, dtype=torch.float32, device=x_norm.device).view(-1,1,1)
    stds  = torch.tensor(MSCLIP_STDS,  dtype=torch.float32, device=x_norm.device).view(-1,1,1)
    x_ref = x_norm * stds + means  # approximate de-norm
    r,g,b = rgb_bands
    arr = x_ref[[r,g,b]].cpu().numpy()
    arr = np.moveaxis(arr, 0, -1)  # HWC

    lo = np.percentile(arr, 1.0); hi = np.percentile(arr, 99.5)
    rgb = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb

def _build_palette(n):
    cmap = plt.get_cmap("tab20")
    cols = []
    for i in range(n):
        cols.append(cmap(i % 20))
    return cols

def load_captions_dictionary_csv(
    captions_dir: str,
    name: str = "*.csv",
    max_phrases: Optional[int] = 50000,
) -> List[str]:
    """
    Load a frequency-sorted concept list from one or more CSVs of the form:

        concept,count
        agricultural field,110948
        waterway,104567
        ...

    Returns a list of unique concept strings (no duplicates), ordered by descending count.
    Respects max_phrases if provided.
    """
    path = str(captions_dir) + str(name)

    all_rows = []

    df = pd.read_csv(path,sep=";")

    col_concept = None
    col_count = None
    for cand in ["concept", "phrase", "caption", "text", "label", "class", "phrase_text","keyword"]:
        if cand in df.columns:
            col_concept = cand
            break
    for cand in ["count", "freq", "frequency", "n", "occurrences"]:
        if cand in df.columns:
            col_count = cand
            break

    if col_concept is None:
        # fallback: first string-like column
        for c in df.columns:
            if df[c].dtype == object:
                col_concept = c
                break
    if col_count is None:
        # fallback: first numeric-like column that's not the concept col
        for c in df.columns:
            if c != col_concept and np.issubdtype(df[c].dtype, np.number):
                col_count = c
                break

    if col_count is None:
        tmp = df[[col_concept]].copy()
        tmp["__count__"] = 1
        col_count = "__count__"
    else:
        tmp = df[[col_concept, col_count]].copy()

    tmp = tmp.dropna(subset=[col_concept])
    tmp[col_concept] = tmp[col_concept].astype(str).str.strip()
    tmp = tmp[tmp[col_concept] != ""]

    tmp_grp = tmp.groupby(col_concept, as_index=False)[col_count].sum()
    tmp_grp = tmp_grp.sort_values(col_count, ascending=False)

    all_rows.append(tmp_grp.rename(columns={col_concept: "concept", col_count: "count"}))

    if not all_rows:
        raise RuntimeError("CSV loader: no valid rows found at all.")

    merged = pd.concat(all_rows, ignore_index=True)

    merged = merged.groupby("concept", as_index=False)["count"].sum()
    merged = merged.sort_values("count", ascending=False)

    phrases = merged["concept"].tolist()

    if max_phrases is not None and len(phrases) > max_phrases:
        phrases = phrases[:max_phrases]

    print(f"[INFO] Loaded {len(phrases)} unique concepts from a CSV file(s).")
    return phrases

def load_captions_dictionary_auto(
    captions_dir: str,
    pattern: str,
    max_phrases: Optional[int] = 50000,
) -> List[str]:
    """
    Decide how to load the caption/phrase corpus based on the pattern/extension.
    - If pattern ends with .csv, we assume a frequency CSV (concept,count).
    - Otherwise we assume parquet caption dumps and fall back to load_captions_dictionary.
    """
    patt_lower = pattern.lower()
    if patt_lower.endswith(".csv"):
        return load_captions_dictionary_csv(
            captions_dir=captions_dir,
            name=pattern,
            max_phrases=max_phrases,
        )
    if patt_lower.endswith("*.csv"):
        return load_captions_dictionary_csv(
            captions_dir=captions_dir,
            name=pattern,
            max_phrases=max_phrases,
        )
    # default: parquet-style caption dumps
    return load_captions_dictionary(
        captions_dir=captions_dir,
        pattern=pattern,
        max_phrases=max_phrases,
    )

def project_patch_tokens_to_text_space(
    ptoks: torch.Tensor,          # [B,P,Dv] 
    pool: torch.Tensor,           # [B,D_img] 
    text_embs: torch.Tensor,      # [N,D_txt] 
    vision_module,                # model.clip_base_model.model.visual
) -> torch.Tensor:
    """
    Return patch embeddings in the SAME space as text_embs (dim D_txt),
    L2-normalized. Handles:
      - ptoks dim already == text dim
      - ptoks dim != text dim, vision_module.proj exists
      - fallback: no proj → tile pooled embedding spatially (keeps us running)

    Output: [B,P,D_txt], float32, normalized.
    """
    B, P, Dv = ptoks.shape
    D_txt = text_embs.shape[-1]  # typically 512

    if Dv == D_txt:
        out = ptoks

    else:

        if pool.ndim == 2:
            pool_tiled = pool.unsqueeze(1).expand(-1, P, -1)  # [B,P,D_txt]
        elif pool.ndim == 3:
            pool_tiled = pool.expand(-1, P, -1)
        else:
            raise RuntimeError(
                f"Unexpected pool shape {pool.shape}; can't tile for fallback."
            )
        out = pool_tiled

    out = F.normalize(out, dim=-1)
    return out  # [B,P,D_txt]

def msclip_denorm_s2_rgb(
    x_norm: torch.Tensor,
    means, stds,
    rgb_bands=(2, 1, 0),     # depends on your ReorderBands; see note below
    ref_max=0.2,
    gamma=1
) -> np.ndarray:
    """
    x_norm: [C,H,W] normalized with MSCLIP stats on S2 scaled-reflectance (reflectance*10000) scale.
    Returns RGB quicklook in [0,1].
    """
    means_t = torch.as_tensor(means, dtype=torch.float32, device=x_norm.device).view(-1, 1, 1)
    stds_t  = torch.as_tensor(stds,  dtype=torch.float32, device=x_norm.device).view(-1, 1, 1)

    # De-norm back to S2 "scaled reflectance" units (~0..10000)
    x_scaled = x_norm * stds_t + means_t

    r, g, b = rgb_bands
    arr = x_scaled[[r, g, b]].detach().cpu().numpy()
    arr = np.moveaxis(arr, 0, -1)  # HWC, still ~0..10000

    arr_ref = arr / 10000.0

    rgb = np.clip(arr_ref / float(ref_max), 0.0, 1.0)

    if gamma and gamma != 1.0:
        rgb = rgb ** (1.0 / float(gamma))

    return rgb

def visualize_positives_tripanel(
    x_pos_btc_hw: torch.Tensor,     # [Bpos,T,C,H,W] or [Bpos,C,H,W]
    top1_pos_bp: torch.Tensor,      # [Bpos,P] (predicted phrase idx per cell) or [Bpos] (single idx)
    y_pos_b1hw: torch.Tensor,       # [Bpos,1,Hc,Wc] (used only to get coarse size; not plotted)
    phrases: list,
    rgb_bands=(2,1,0),
    savepath: Optional[str] = None,
    alpha: float = 0.35,
    max_legend: int = 15,
    min_frac: float = 0.02,
    title_prefix: str = "",
):
    """
    New layout (no GT panel):
      Col 0: high-res RGB (for context)
      Col 1: coarse predicted phrase overlay on coarse RGB (same Hc x Wc grid)
      Col 2: legend (bigger font, more spacing, wrapped phrases)

    Notes:
      - We only use y_pos_b1hw to determine coarse Hc x Wc.
      - No ground-truth overlay is drawn anymore.
    """

    # normalize input dims to [B,T,C,H,W]
    if x_pos_btc_hw.ndim == 4:
        x_pos_btc_hw = x_pos_btc_hw.unsqueeze(1)
    Bpos, T, C, H, W = x_pos_btc_hw.shape

    # coarse size from labels (not shown, just used for sizing)
    _, _, Hc, Wc = y_pos_b1hw.shape

    palette = _build_palette(256)

    # legend styling tweaks
    legend_fontsize = 12       # bigger text
    legend_line_step = 0.16    # more vertical spacing
    legend_patch_w = 0.12
    legend_patch_h = 0.10
    wrap_line_len = 46         # wrap a bit wider since we widened the legend column
    wrap_max_lines = 5

    for b in range(Bpos):
        # ---- RGB hi-res (col 0)
        img_full = x_pos_btc_hw[b, 0]                         # [C,H,W] (MS-CLIP-normalized)
        rgb_full_disp = msclip_denorm_s2_rgb(
            img_full, means=MSCLIP_MEANS, stds=MSCLIP_STDS
        )

        # ---- coarse RGB for overlay (col 1)
        img_full_bchw = img_full.unsqueeze(0)                 # [1,C,H,W]
        img_coarse = torch.nn.functional.interpolate(
            img_full_bchw, size=(Hc, Wc), mode="bilinear", align_corners=False
        )[0]                                                  # [C,Hc,Wc]
        rgb_coarse_disp = msclip_denorm_s2_rgb(
            img_coarse, means=MSCLIP_MEANS, stds=MSCLIP_STDS
        )

        # ---- predicted label grid @ coarse res
        if top1_pos_bp.ndim == 1:
            label_grid = torch.full(
                (Hc, Wc),
                int(top1_pos_bp[b].item()),
                dtype=torch.long,
                device=x_pos_btc_hw.device
            )
        else:
            Pcur = top1_pos_bp.shape[1]
            if Pcur == Hc * Wc:
                label_grid = top1_pos_bp[b].view(Hc, Wc).to(torch.long)
            else:
                side = int(round(Pcur ** 0.5))
                tmp_grid = top1_pos_bp[b].view(side, side).to(torch.long)  # [side,side]
                tmp_resized = torch.nn.functional.interpolate(
                    tmp_grid[None, None].float(), size=(Hc, Wc), mode="nearest"
                )[0, 0].to(torch.long)
                label_grid = tmp_resized

        uniq, counts = torch.unique(label_grid, return_counts=True)
        freqs = counts.float() / float(Hc * Wc)
        keep_pairs = [(int(u.item()), float(f.item())) for u, f in zip(uniq, freqs)]
        keep_pairs = sorted(keep_pairs, key=lambda x: x[1], reverse=True)
        keep_pairs = [kp for kp in keep_pairs if kp[1] >= min_frac][:max_legend]

        # RGBA overlay from kept labels only
        pred_overlay = np.zeros((Hc, Wc, 4), dtype=np.float32)
        label_grid_np = label_grid.detach().cpu().numpy()
        uniq_all = np.unique(label_grid_np)

        # paint all patches
        for k_label in uniq_all:
            color_rgb = palette[int(k_label) % len(palette)]
            mask_k = (label_grid_np == k_label)
            pred_overlay[mask_k, 0] = color_rgb[0]
            pred_overlay[mask_k, 1] = color_rgb[1]
            pred_overlay[mask_k, 2] = color_rgb[2]
            pred_overlay[mask_k, 3] = alpha

        # ---- Figure: 1 row, 3 columns (RGB | Pred overlay | Legend)
        fig = plt.figure(figsize=(14.5, 4.8), dpi=150)
        gs = fig.add_gridspec(1, 3, width_ratios=[5, 5, 4.5], wspace=0.18)

        # Col 0: RGB hi-res
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_rgb.imshow(rgb_full_disp)
        ax_rgb.set_title(f"{title_prefix} sample {b} – RGB", fontsize=10)
        ax_rgb.axis("off")

        # Col 1: predicted overlay (coarse)
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_pred.imshow(rgb_coarse_disp, interpolation="nearest")
        ax_pred.imshow(pred_overlay, interpolation="nearest")
        ax_pred.set_title("Predicted regions (coarse)", fontsize=10)
        ax_pred.axis("off")

        # Col 2: legend (bigger)
        ax_leg = fig.add_subplot(gs[0, 2])
        ax_leg.axis("off")
        y_cursor = 0.96
        dy = legend_line_step

        for k_label, frac in keep_pairs:
            if y_cursor < 0.06:
                break
            phrase_full = phrases[k_label] if 0 <= k_label < len(phrases) else f"id {k_label}"
            phrase_wrapped = _wrap_phrase(phrase_full, max_len_line=wrap_line_len, max_lines=wrap_max_lines)

            color_rgb = palette[k_label % len(palette)]
            # color patch
            ax_leg.add_patch(
                plt.Rectangle(
                    (0.05, y_cursor - legend_patch_h * 0.5),
                    legend_patch_w, legend_patch_h,
                    transform=ax_leg.transAxes,
                    color=color_rgb,
                )
            )
            # text
            ax_leg.text(
                0.20, y_cursor,
                f"{(frac*100):.1f}%\n{phrase_wrapped}",
                transform=ax_leg.transAxes,
                fontsize=legend_fontsize,
                va="top",
            )
            y_cursor -= dy

        # Save/show
        if savepath is not None:
            base, ext = os.path.splitext(savepath)
            out_path = f"{base}_b{b:02d}{ext if ext else '.png'}"
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
            print(f"[INFO] Wrote {out_path}")
            plt.close(fig)
        else:
            plt.show()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH))
def main(cfg: DictConfig):

    cfg = OmegaConf.to_container(cfg, resolve=True)

    set_seed(cfg["SET-UP"]["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    for caption in cfg["captions_glob"]:
        out_name = os.path.splitext(caption)[0]
        print("[INFO] Loading MS-CLIP…")
        model_, preprocess, tokenizer = build_model(
            model_name=cfg["MODEL"]["msclip_model_name"],
            pretrained=cfg["MODEL"]["pretrained"],
            ckpt_path=cfg["MODEL"]["msclip_ckpt"],
            device=cfg["device"],
            channels=cfg["MODEL"]["input_dim"],
        )
        model = model_.to(cfg["device"]).eval()
    
        print(f"[INFO] Loading HF parquet dataloader from", cfg["data_dir"]," split=",cfg["split"])
        #loader = get_data(cfg).test_dataloader()
        loader = get_data(cfg).val_dataloader()

        print("[INFO] Loading captions corpus…")
        phrases = load_captions_dictionary_auto(
            cfg["captions_dir"],
            caption,
            cfg["max_phrases"],
        )

        if False:
            model = get_model(cfg,cfg["device"])
            
        ckpt = cfg["CHECKPOINT"].get("load_from_checkpoint", "")
        if ckpt:
            load_from_checkpoint(model, ckpt)
        model.eval().to(cfg["device"])

        print("[INFO] Encoding text embeddings…")
        text_embs = tokenize_phrases(tokenizer, phrases, cfg["text_batch_size"], cfg["device"], model)  # [N,D]

        baseline_phrase = "a satellite view of"
        baseline_text = tokenize_phrases(tokenizer, [baseline_phrase], 1, cfg["device"], model)[0]  # [D], normalized

        sum_unit = None      
        sum_raw = None      
        n_tokens = 0
        sum_cos = 0.0        


        global_counts = np.zeros(len(phrases), dtype=int)

        # new: sum of cosine sims for that phrase over all assignments
        global_sim_sums = np.zeros(len(phrases), dtype=float)

        # optional (not strictly needed, but nice to have if you want max confidence later)
        global_sim_max  = np.full(len(phrases), -1e9, dtype=float)
        plotted_batches = 0

        image_entropies = []

        vision = model.clip_base_model.model.visual  
        vision.output_tokens = True

        model_config = cfg["MODEL"]
        if model_config is not None and "clearclip" in model_config and model_config["clearclip"]["enabled"]:
            num_patched = maybe_patch_clearclip(model.image_encoder, model_config["clearclip"])
            if num_patched > 0:
                print(f"[ClearCLIP] Patched last {num_patched} vision blocks "
                    f"(keep_ffn={model_config['clearclip'].get('keep_ffn', False)}, "
                    f"keep_residual={model_config['clearclip'].get('keep_residual', False)})")

        if model_config is not None and "sclip" in model_config and model_config["sclip"]["enabled"]:
            num_patched = maybe_patch_sclip(model.image_encoder, model_config["sclip"])
            if num_patched > 0:
                print(f"[SCLIP] Patched last {num_patched} vision blocks "
                        f"(CSA attention)")

        for bi, batch in enumerate(tqdm(loader)):
            # batch[0] is the dict with "inputs" if you are using the test set
            # batch is the dict with "inputs" if you are using the val set

            input_ = batch
            x = input_["inputs"]             # [B,T,C,H,W], MS-CLIP normed
            B,T,C,H,W = x.shape
            x = x.to(cfg["device"], non_blocking=True)

            patch_tokens_over_time = []
            pooled_over_time = []

            X = x[:, 0,:,:,:]                    # [B,C,H,W]

            with torch.inference_mode():
                pool, ptoks = model.image_encoder(X)
                ptoks = vision.ln_post(ptoks)      # [B*T, P, 768] -> LN
                ptoks = ptoks @ vision.proj

            Bcur, Pcur, Dcur = ptoks.shape

            ptoks_f = ptoks.float()

            # normalized patches for cosine baseline
            ptoks_unit = F.normalize(ptoks_f, dim=-1)  # [B,P,D]

            # init sums once
            if sum_unit is None:
                sum_unit = torch.zeros((Dcur,), device=ptoks.device, dtype=torch.float64)
                sum_raw  = torch.zeros((Dcur,), device=ptoks.device, dtype=torch.float64)

            sum_unit += ptoks_unit.double().sum(dim=(0, 1))
            sum_raw  += ptoks_f.double().sum(dim=(0, 1))
            n_tokens += Bcur * Pcur


            sum_cos += float((ptoks_unit * baseline_text.view(1, 1, -1)).sum(dim=-1).sum().item())


            # cosine sim to all phrases, per patch
            scores, idx = assign_labels_to_patches(ptoks, text_embs, topk=1)
            # [B,P,1] →  [B,P]
            top1 = idx.squeeze(-1)
            sims = scores.squeeze(-1)

            # --- accumulate global stats for CSV ---
            top1_np = top1.detach().cpu().numpy()
            sims_np = sims.detach().cpu().numpy()

            Bcur, Pcur = top1_np.shape

            for bb in range(Bcur):
                patch_labels = top1_np[bb]  # shape [Pcur]
                # count how many times each phrase appears within this image
                counts_img = np.bincount(patch_labels, minlength=len(phrases))

                if Pcur > 0:
                    probs_img = counts_img[counts_img > 0].astype(np.float64) / float(Pcur)
                    # Shannon entropy (nats); lower = more uniform / less noisy labels per image
                    H_img = -np.sum(probs_img * np.log(probs_img))
                    image_entropies.append(H_img)

            for bb in range(Bcur):
                for pp in range(Pcur):
                    k = int(top1_np[bb, pp])
                    sim_val = float(sims_np[bb, pp])

                    global_counts[k] += 1
                    global_sim_sums[k] += sim_val
                    if sim_val > global_sim_max[k]:
                        global_sim_max[k] = sim_val

            # --- positive filtering for visualization ---
            pos_idx_list = _get_positive_indices(input_, min_pos_frac=0.3)

            # safety stop condition you added
            if plotted_batches > cfg["n_show_batches"]:
                sys.exit(0)

            if len(pos_idx_list) > 0 and plotted_batches < cfg["n_show_batches"]:

                # slice to positives only
                x_pos    = x[pos_idx_list]        # [Bpos,T,C,H,W]
                top1_pos = top1[pos_idx_list]     # [Bpos,P]

                # get binarized GT label mask at coarse resolution
                # NOTE: your dataloader stores labels in input_["labels"], shape [B,Hc,Wc,C?] or similar
                # you used .permute(0,3,1,2) earlier, so let's keep that
                y_full = input_["labels"]
                y_bin  = _binarize_labels(y_full, input_["inputs"])  # -> [B,1,Hc,Wc] or None

                if y_bin is not None:
                    y_pos = y_bin[pos_idx_list].to(x.device)

                    savepath = os.path.join(cfg["out_dir"], f"batch_{bi:04d}_positives_{out_name}.png")
                    visualize_positives_tripanel(
                        x_pos_btc_hw = x_pos,
                        top1_pos_bp  = top1_pos,
                        y_pos_b1hw   = y_pos,
                        phrases      = phrases,
                        rgb_bands    = cfg["rgb_bands"],
                        savepath     = savepath,
                        alpha        = 0.35,
                        max_legend   = 15,
                        min_frac     = 0.02,
                        title_prefix = f"{cfg['split']}",
                    )
                    plotted_batches += 1

        # ---- finalize mu + cosine checks ----
        if n_tokens > 0:
            mean_unit = (sum_unit / n_tokens).float()   # mean of unit patch vectors (not unit itself)
            mean_raw  = (sum_raw  / n_tokens).float()   # mean of raw patch embeddings

            avg_cos = sum_cos / n_tokens  # exactly mean_i cos(p_i, phrase)

            cos_mu_unit = float((F.normalize(mean_unit, dim=0) * baseline_text).sum().item())
            cos_mu_raw  = float((F.normalize(mean_raw,  dim=0) * baseline_text).sum().item())

            print("\n[MU CHECK]")
            print(f"  phrase = {baseline_phrase!r}")
            print(f"  tokens counted = {n_tokens}")
            print(f"  avg_cos_over_patches           = {avg_cos:.6f}")
            print(f"  cos(normalize(mean_unit), t)   = {cos_mu_unit:.6f}")
            print(f"  cos(normalize(mean_raw),  t)   = {cos_mu_raw:.6f}")
        else:
            print("[MU CHECK] No tokens counted; check dataloader / early breaks.")

        # Global plot adn CSV
        topN = cfg["topk_words_plot"]
        top_idx = np.argsort(-global_counts)[:topN]
        top_vals = global_counts[top_idx]
        top_labels = [phrases[i] for i in top_idx]

        if False:
            plt.figure(figsize=(10, max(4, topN*0.25)))
            plt.barh(range(len(top_labels)-1, -1, -1), top_vals[::-1])
            plt.yticks(range(len(top_labels)-1, -1, -1), top_labels[::-1], fontsize=7)
            plt.xlabel("Patch count (top-1)")
            plt.title(f"Top {topN} phrases across dataset (by patch top-1 label)")
            bar_path = os.path.join(cfg["out_dir"], f"global_top_{topN}_phrases_{out_name}.png")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=200)
            plt.close()
            print(f"[INFO] Wrote {bar_path}")

        # avoid divide-by-zero: where count == 0, avg_sim should be NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_sims = np.where(global_counts > 0,
                                global_sim_sums / global_counts,
                                np.nan)

        df = pd.DataFrame({
            "phrase": phrases,
            "count": global_counts,
            "avg_cosine_sim": avg_sims,
            "max_cosine_sim": global_sim_max
        })

        # sort by count first, then avg cosine similarity
        df = df.sort_values(["count", "avg_cosine_sim"], ascending=[False, False])

        csv_path = os.path.join(cfg["out_dir"], f"global_phrase_counts_{out_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Wrote {csv_path}")

        total_patches = int(global_counts.sum())
        active_phrases = int((global_counts > 0).sum())

        if total_patches > 0 and active_phrases > 0:
            # Global phrase-usage entropy over all patches in the dataset
            probs_global = global_counts[global_counts > 0].astype(np.float64) / float(total_patches)
            H_global_nats = float(-np.sum(probs_global * np.log(probs_global)))
            H_global_bits = H_global_nats / np.log(2.0)
            H_global_norm = H_global_nats / np.log(active_phrases)  # normalized to [0,1]

            num_images = len(image_entropies)
            if num_images > 0:
                image_entropies = np.asarray(image_entropies, dtype=np.float64)
                H_image_mean = float(image_entropies.mean())
                H_image_std  = float(image_entropies.std())
                H_image_min  = float(image_entropies.min())
                H_image_max  = float(image_entropies.max())
            else:
                H_image_mean = H_image_std = H_image_min = H_image_max = np.nan

            print("[INFO] Entropy over global phrase usage (patch labels)")
            print(f"       H_global (nats)       = {H_global_nats:.4f}")
            print(f"       H_global (bits)       = {H_global_bits:.4f}")
            print(f"       H_global_norm (0-1)   = {H_global_norm:.4f}")
            print("[INFO] Mean per-image label entropy (nats): "
                f"{H_image_mean:.4f} ± {H_image_std:.4f} "
                f"(min={H_image_min:.4f}, max={H_image_max:.4f})")

            # Save a tiny CSV so you can compare models easily
            entropy_df = pd.DataFrame({
                "total_patches": [total_patches],
                "active_phrases": [active_phrases],
                "H_global_nats": [H_global_nats],
                "H_global_bits": [H_global_bits],
                "H_global_normalized": [H_global_norm],
                "num_images": [num_images],
                "H_image_mean_nats": [H_image_mean],
                "H_image_std_nats": [H_image_std],
                "H_image_min_nats": [H_image_min],
                "H_image_max_nats": [H_image_max],
            })

            entropy_path = os.path.join(cfg["out_dir"], f"entropy_stats_{out_name}.csv")
            entropy_df.to_csv(entropy_path, index=False)
            print(f"[INFO] Wrote {entropy_path}")
        else:
            print("[WARN] Could not compute entropy (no patches or no active phrases).")




if __name__ == "__main__":
    main()
