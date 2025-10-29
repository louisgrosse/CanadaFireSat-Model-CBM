
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from src.data import get_data

from src.data.Canada.data_transforms import Canada_segmentation_transform, MSCLIP_MEANS, MSCLIP_STDS

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model


@dataclass
class Config:
    # MS-CLIP
    model_name: str = "Llama3-MS-CLIP-Base"
    pretrained: bool = True
    ckpt_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    channels: int = 10
    # Data
    bands_mean:str =  "/home/louis/Code/data/metadata/bands_mean.json",
    bands_std:str =  "/home/louis/Code/data/metadata/bands_std.json",
    data_dir:str =  "/home/louis/Code/wildfire-forecast/hug_data",
    split: str = "train"                  # "train" | "validation" | "test" | "test_hard"
    batch_size: int = 8
    num_workers: int = 4
    # Transforms (match your training setup)
    input_img_res: int = 224
    img_res: int = 224
    out_H: int = 25
    out_W: int = 25
    train_max_seq_len: int = 4            # use a small T for speed if desired
    val_max_seq_len: int = 4
    use_msclip_norm: bool = True          # important: keep MS-CLIP stats
    with_doy: bool = True
    with_loc: bool = False
    ds_labels: bool = True                # dataset provides segmentation labels
    # Dictionary
    captions_dir: str = "/home/louis/Code/wildfire-forecast/dictionnary/"
    captions_glob: str = "*.parquet"
    max_phrases: Optional[int] = 50000
    text_batch_size: int = 512
    # Visualization
    rgb_bands: Tuple[int,int,int] = (2,1,0)  # (B4,B3,B2)-like indices in MS-CLIP 10-band order
    n_show_batches: int = 2                  # how many batches to visualize
    topk_words_plot: int = 25
    seed: int = 42
    # Output
    out_dir: str = "results/dictionnary/patch_naming_hf"

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_captions_dictionary(captions_dir: str, pattern: str = "*.parquet",
                             max_phrases: Optional[int] = 50000) -> List[str]:
    paths = sorted(glob.glob(os.path.join(captions_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {captions_dir!r} with pattern {pattern!r}")
    phrases: List[str] = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue
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
    if max_phrases is not None and len(uniq) > max_phrases:
        uniq = uniq[:max_phrases]
    print(f"[INFO] Loaded {len(uniq)} unique phrases from {len(paths)} parquet file(s).")
    return uniq

def tokenize_phrases(tokenizer, phrases: List[str], batch_size: int, device: str, model) -> torch.Tensor:
    
    embs = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        z = model.encode_text(tokens)
        z = F.normalize(z, dim=-1)
        embs.append(z)
    return torch.cat(embs, dim=0)

def assign_labels_to_patches(patch_embs: torch.Tensor, text_embs: torch.Tensor, topk: int = 1):
    sims = patch_embs @ text_embs.t()
    return torch.topk(sims, k=topk, dim=-1)

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

def visualize_batch_with_patch_labels(x_btchw: torch.Tensor,
                                      labels_grid_bp: torch.Tensor,
                                      phrases: List[str],
                                      rgb_bands=(3,2,1),
                                      savepath: Optional[str] = None,
                                      title_prefix: str = ""):
    """x_btchw: [B,T,C,H,W] normalized
       labels_grid_bp: [B,P] top-1 phrase idx per patch (flattened P=14*14)
    """
    B,T,C,H,W = x_btchw.shape
    Hp = Wp = H // 16  # 224/16=14
    ph = H // Hp; pw = W // Wp

    num_to_plot = min(B, 4)
    fig, axs = plt.subplots(num_to_plot, 1, figsize=(7, 7*num_to_plot))
    if num_to_plot == 1: axs = [axs]

    for b in range(num_to_plot):

        x_chw = x_btchw[b,0]  # [C,H,W]
        rgb = msclip_denorm_to_rgb(x_chw, rgb_bands=rgb_bands)

        ax = axs[b]
        ax.imshow(rgb)
        top1 = labels_grid_bp[b].view(Hp, Wp).cpu().numpy()

        for i in range(Hp):
            for j in range(Wp):
                y = i*ph + ph//2; x = j*pw + pw//2
                idx = int(top1[i,j])
                text = phrases[idx] if 0 <= idx < len(phrases) else "?"
                ax.text(x, y, text, ha="center", va="center", fontsize=5, alpha=0.85,
                        bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.1"))
        ax.set_title(f"{title_prefix} sample {b}")
        ax.axis("off")

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def main(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("[INFO] Loading MS-CLIP…")
    model_, preprocess, tokenizer = build_model(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        ckpt_path=cfg.ckpt_path,
        device=cfg.device,
        channels=cfg.channels,
    )
    model = model_.to(cfg.device).eval()

    model_config = dict(
        input_img_res=cfg.input_img_res,
        img_res=cfg.img_res,
        out_H=cfg.out_H,
        out_W=cfg.out_W,
        train_max_seq_len=cfg.train_max_seq_len,
        val_max_seq_len=cfg.val_max_seq_len,
        ds_labels=cfg.ds_labels,
    )
    transform = Canada_segmentation_transform(
        model_config=model_config,
        mean_file="", std_file="",
        is_training=False,
        is_eval=False,
        use_msclip_norm=cfg.use_msclip_norm,
        with_doy=cfg.with_doy,
        with_loc=cfg.with_loc,
        bands=None,
    )

   
    print(f"[INFO] Loading HF parquet dataloader from {cfg.data_dir} split={cfg.split}")
    loader = get_data(
        data_dir=cfg.data_dir,
        transform=transform,
        split="val" if cfg.split in ("validation", "val") else cfg.split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        my_collate=None,
        with_loc=cfg.with_loc,
    )


    print("[INFO] Loading captions corpus…")
    phrases = load_captions_dictionary(cfg.captions_dir, cfg.captions_glob, cfg.max_phrases)

    print("[INFO] Encoding text embeddings…")
    text_embs = tokenize_phrases(tokenizer, phrases, cfg.text_batch_size, cfg.device, model)  # [N,D]

    global_counts = np.zeros((len(phrases),), dtype=np.int64)
    plotted_batches = 0

    for bi, batch in enumerate(loader):
        x = batch["inputs"]  # [B,T,C,H,W] normalized already to MS-CLIP stats
        B,T,C,H,W = x.shape
        x = x.to(cfg.device, non_blocking=True)

        patch_tokens_over_time = []
        for t in range(T):
            X = x[:, t]  # [B,C,H,W] 
            ptoks = model.image_encoder(X)  # [B,P,D]
            patch_tokens_over_time.append(ptoks)


        stk = torch.stack(patch_tokens_over_time, dim=0)  # [T,B,P,D]
        ptoks = stk.mean(dim=0)  # [B,P,D]

 
        scores, idx = assign_labels_to_patches(ptoks, text_embs, topk=1)  # each [B,P,1]
        top1 = idx.squeeze(-1)  # [B,P]

        for b in range(top1.shape[0]):
            ids = top1[b].detach().cpu().numpy()
            for k in ids:
                global_counts[int(k)] += 1


        if plotted_batches < cfg.n_show_batches:
            savepath = os.path.join(cfg.out_dir, f"batch_{bi:04d}_patch_labels.png")
            visualize_batch_with_patch_labels(batch["inputs"], top1, phrases, rgb_bands=cfg.rgb_bands,
                                              savepath=savepath, title_prefix=f"split={cfg.split}")
            print(f"[INFO] Wrote {savepath}")
            plotted_batches += 1

    # Global plot & CSV
    topN = cfg.topk_words_plot
    top_idx = np.argsort(-global_counts)[:topN]
    top_vals = global_counts[top_idx]
    top_labels = [phrases[i] for i in top_idx]

    plt.figure(figsize=(10, max(4, topN*0.25)))
    plt.barh(range(len(top_labels)-1, -1, -1), top_vals[::-1])
    plt.yticks(range(len(top_labels)-1, -1, -1), top_labels[::-1], fontsize=7)
    plt.xlabel("Patch count (top-1)")
    plt.title(f"Top {topN} phrases across dataset (by patch top-1 label)")
    bar_path = os.path.join(cfg.out_dir, f"global_top_{topN}_phrases.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[INFO] Wrote {bar_path}")

    df = pd.DataFrame({"phrase": phrases, "count": global_counts}).sort_values("count", ascending=False)
    csv_path = os.path.join(cfg.out_dir, "global_phrase_counts.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote {csv_path}")

if __name__ == "__main__":
    cfg = Config()
    # cfg.data_dir = "/path/to/hf/parquet_root"
    # cfg.captions_dir = "data/captions"
    main(cfg)
