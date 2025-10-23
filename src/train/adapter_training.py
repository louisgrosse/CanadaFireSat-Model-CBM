
# adapter_training.py
# --------------------
# Train lightweight adapters (embedding- or pixel-space) to align L1C to L2A for MS-CLIP.
#
# Usage (example):
#   python adapter_training.py     #     --train_dir /path/to/worldstrat/train_parquets     #     --val_dir /path/to/worldstrat/val_parquets     #     --msclip_model_name ViT-B/16     #     --msclip_ckpt /path/to/msclip.ckpt     #     --bands 10     #     --band_order 0,1,2,3,4,5,6,7,8,9     #     --means 0.0366,0.0393,0.0293,0.0367,0.0568,0.0727,0.1161,0.1985,0.2361,0.1925     #     --stds  0.0515,0.0537,0.0464,0.0566,0.0749,0.0879,0.1123,0.1320,0.1301,0.1214     #     --model_type embed_dlr     #     --pool token     #     --epochs 20     #     --rank 16     #     --batch_size 16     #     --num_workers 4     #     --save_dir ./checkpoints
#
# Notes:
# - Expects MS-CLIP to be importable with build_model() that returns an object with
#   .image_encoder.get_patch_embeddings(x) -> [B, Tokens, C].
# - The dataset expects Parquet columns like: 10x_L1C, 20x_L1C, 10x_L2A, 20x_L2A (pickled numpy arrays).
# - Set --means/--stds to the MS-CLIP channel stats used during pretraining (per band, [0,1] reflectance scale).
#
from __future__ import annotations

import os
import sys
import glob
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
from src.data.utils import extract_stats


from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm import tqdm
import pyarrow.parquet as pq
import json
import hydra
from src.constants import CONFIG_PATH
from src.constants import MSCLIP_ORDER_10


from src.models.adapters import build_adapter

# Import MS-CLIP build util (user environment should provide this)
sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model as build_msclip


import torch
import torch.nn.functional as F
from collections import defaultdict

@torch.no_grad()
def compute_invariance_metrics(encoder, loader, mean_t, std_t, pool="token", max_batches=100, device=None):
    """
    Aggregates dataset-wide L1C vs L2A metrics:
      - patch norms (mean±std per domain)
      - cosine (mean±std,min,max) per token
      - embedding MAE / MAX
      - norm ratio stats
      - reflectance out-of-range fractions (>1)
    Assumes each batch yields (l1c, l2a) in reflectance space before MS-CLIP normalization.
    """
    device = device or next(encoder.parameters()).device
    M = defaultdict(list)
    seen = 0

    def _norms(z):
        # z: [B,T,C] or [B,C] if pooled
        return z.norm(dim=-1).reshape(-1)

    for l1c, l2a in loader:
        if seen >= max_batches: break
        seen += 1

        # Move + normalize (clone to avoid aliasing)
        l1c = ((l1c.to(device) - mean_t) / std_t).contiguous().clone()
        l2a = ((l2a.to(device) - mean_t) / std_t).contiguous().clone()

        # Reflectance out-of-range fractions (before norm reconstruction)
        l1c_ref = l1c * std_t + mean_t
        l2a_ref = l2a * std_t + mean_t
        M["frac_gt1_l1c"].append((l1c_ref > 1).float().mean().item())
        M["frac_gt1_l2a"].append((l2a_ref > 1).float().mean().item())

        # Embeddings
        z1 = encoder.get_patch_embeddings(l1c)
        z2 = encoder.get_patch_embeddings(l2a)
        if pool == "mean":
            z1 = z1.mean(dim=1)  # [B,C]
            z2 = z2.mean(dim=1)
        # token path stays [B,T,C]

        # Norms per domain
        M["norm_l1c"].append(_norms(z1).mean().item())
        M["norm_l2a"].append(_norms(z2).mean().item())

        # Cosine per token
        z1n = F.normalize(z1, dim=-1)
        z2n = F.normalize(z2, dim=-1)
        cos = (z1n * z2n).sum(dim=-1).reshape(-1)
        M["cos_mean"].append(cos.mean().item())
        M["cos_std"].append(cos.std().item())
        M["cos_min"].append(cos.min().item())
        M["cos_max"].append(cos.max().item())

        # Embedding difference
        diff = (z1 - z2).reshape(-1, z1.shape[-1]).abs()
        M["mae"].append(diff.mean().item())
        M["max"].append(diff.max().item())

        # Norm ratio
        r = (_norms(z1) / (_norms(z2) + 1e-8))
        M["ratio_mean"].append(r.mean().item())
        M["ratio_std"].append(r.std().item())
        M["ratio_min"].append(r.min().item())
        M["ratio_max"].append(r.max().item())

    # Aggregate
    agg = {k: (sum(v)/len(v) if len(v)>0 else float("nan")) for k,v in M.items()}
    return agg

def latex_invariance_table(agg, title="MS-CLIP L1C↔L2A invariance (dataset-wide)"):
    def f(x, p=6): return f"{x:.{p}f}"
    lines = []
    lines += [r"\begin{table*}[t]", r"\centering", rf"\caption{{{title}}}", r"\label{tab:msclip_invariance_wide}",
              r"\begin{tabular}{lccc}", r"\toprule",
              r"\textbf{Metric} & \textbf{L1C} & \textbf{L2A} & \textbf{L1C vs L2A} \\",
              r"\midrule"]
    lines += [rf"Patch norm (mean) & {f(agg['norm_l1c'],3)} & {f(agg['norm_l2a'],3)} & -- \\"]
    lines += [rf"Cosine (per token) & -- & -- & {f(agg['cos_mean'],6)} $\pm$ {f(agg['cos_std'],6)} "
              rf"[min {f(agg['cos_min'],6)}, max {f(agg['cos_max'],6)}] \\"]
    lines += [rf"Embedding diff (MAE / MAX) & -- & -- & {f(agg['mae'],6)} / {f(agg['max'],6)} \\"]
    lines += [rf"Norm ratio $||z_{{L1C}}||/||z_{{L2A}}||$ & -- & -- & {f(agg['ratio_mean'],4)} "
              rf"$\pm$ {f(agg['ratio_std'],4)} [min {f(agg['ratio_min'],4)}, max {f(agg['ratio_max'],4)}] \\"]
    lines += [rf"Reflectance frac $>1$ & {f(agg['frac_gt1_l1c'],6)} & {f(agg['frac_gt1_l2a'],6)} & -- \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)

def read_vector_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    if p.suffix.lower() == '.npy':
        return [float(x) for x in np.load(p).tolist()]
    if p.suffix.lower() in ('.json', '.jsn'):
        with open(p, 'r') as f:
            arr = json.load(f)
        if isinstance(arr, dict):
            arr = next(iter(arr.values()))
        return [float(x) for x in arr]
    with open(p, 'r') as f:
        text = f.read().strip()
    tokens = [t for t in text.replace(',', ' ').split() if t]
    return [float(t) for t in tokens]

def first_param_device(module: nn.Module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    return torch.device("cpu")

def move_to_device(module: nn.Module, device: torch.device):
    # Move the module and common nested attributes used by MS-CLIP
    module.to(device)
    for attr in ("image_encoder", "visual", "model", "backbone"):
        if hasattr(module, attr):
            child = getattr(module, attr)
            if isinstance(child, nn.Module):
                child.to(device)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)

def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = safe_normalize(a, dim=-1)
    b = safe_normalize(b, dim=-1)
    a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=1.0, neginf=-1.0)
    return (1.0 - (a * b).sum(dim=-1)).mean()

def l2_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b)

def coral_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    n_a = max(a.shape[0]-1, 1)
    n_b = max(b.shape[0]-1, 1)
    cov_a = (a.T @ a) / n_a
    cov_b = (b.T @ b) / n_b
    return F.mse_loss(cov_a, cov_b)

class WorldStratPairsStream(IterableDataset):
    REQUIRED_COLS = ("10x_L1C", "20x_L1C", "10x_L2A", "20x_L2A")

    def __init__(self, parquet_dir: str, band_order, input_hw=(264,264), out_hw=(224,224), io_batch_size: int = 64):
        super().__init__()
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        assert files, f"No parquet files found in {parquet_dir}"
        self.files = files
        self.band_order = band_order
        self.input_hw = input_hw
        self.out_hw = out_hw
        self.io_batch_size = max(1, int(io_batch_size))

    @staticmethod
    def _to_reflectance01(x: torch.Tensor) -> torch.Tensor:
        xmax = float(x.max().item()) if x.numel() else 0.0
        if xmax <= 1.5:
            return x
        if xmax <= 255.0:
            return x / 255.0
        return x / 10000.0

    def _row_to_sample(self, row_dict):
        ten_L1C    = torch.from_numpy(pickle.loads(row_dict["10x_L1C"]).copy()).float()
        twenty_L1C = torch.from_numpy(pickle.loads(row_dict["20x_L1C"]).copy()).float()
        ten_L2A    = torch.from_numpy(pickle.loads(row_dict["10x_L2A"]).copy()).float()
        twenty_L2A = torch.from_numpy(pickle.loads(row_dict["20x_L2A"]).copy()).float()

        twenty_L1C = F.interpolate(twenty_L1C.unsqueeze(0), size=ten_L1C.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        twenty_L2A = F.interpolate(twenty_L2A.unsqueeze(0), size=ten_L2A.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

        x_L1C = torch.cat([ten_L1C, twenty_L1C], dim=0)[self.band_order, ...]
        x_L2A = torch.cat([ten_L2A, twenty_L2A], dim=0)[self.band_order, ...]

        x_L1C = self._to_reflectance01(x_L1C)
        x_L2A = self._to_reflectance01(x_L2A)

        if self.input_hw is not None:
            x_L1C = F.interpolate(x_L1C.unsqueeze(0), size=self.input_hw, mode="bilinear", align_corners=False).squeeze(0)
            x_L2A = F.interpolate(x_L2A.unsqueeze(0), size=self.input_hw, mode="bilinear", align_corners=False).squeeze(0)

        x_L1C = F.interpolate(x_L1C.unsqueeze(0), size=self.out_hw, mode="bilinear", align_corners=False).squeeze(0)
        x_L2A = F.interpolate(x_L2A.unsqueeze(0), size=self.out_hw, mode="bilinear", align_corners=False).squeeze(0)
        return x_L1C, x_L2A

    def _iter_one_file(self, file_path: str):
        pf = pq.ParquetFile(file_path, memory_map=True)
        for batch in pf.iter_batches(batch_size=self.io_batch_size, columns=list(self.REQUIRED_COLS), use_threads=False):
            col_idx = {name: i for i, name in enumerate(batch.schema.names)}
            cols = {name: batch.columns[col_idx[name]] for name in self.REQUIRED_COLS}
            for i in range(batch.num_rows):
                try:
                    row = {name: cols[name][i].as_py() for name in self.REQUIRED_COLS}
                    yield self._row_to_sample(row)
                except Exception as e:
                    print(f"[WARN] Skipping row {i} in {file_path}: {e}")

    def __iter__(self):
        wi = get_worker_info()
        if wi is None:
            file_iter = self.files
        else:
            file_iter = [f for idx, f in enumerate(self.files) if idx % wi.num_workers == wi.id]
        for f in file_iter:
            yield from self._iter_one_file(f)

def make_norm_tensors(means, stds, device):
    mean = torch.tensor(means, dtype=torch.float32, device=device).view(-1,1,1)
    std  = torch.tensor(stds,  dtype=torch.float32, device=device).view(-1,1,1)
    return mean, std

def normalize_tensor(x, mean, std):
    C = x.shape[0]
    return (x - mean[:C]) / std[:C]

def get_embeddings(encoder, x, pool="token", need_grad=False):
    with torch.set_grad_enabled(need_grad):
        patches = encoder.get_patch_embeddings(x)
        if pool == "token":
            return patches
        elif pool == "mean":
            return patches.mean(dim=1)
        else:
            raise ValueError("MODEL.pool must be 'token' or 'mean'")

def bandwise_stats(x: torch.Tensor, qs=(1,50,99)):
    B, C, H, W = x.shape
    flat = x.permute(1,0,2,3).contiguous().view(C, -1)
    mins = flat.min(dim=1).values
    maxs = flat.max(dim=1).values
    means = flat.mean(dim=1)
    stds = flat.std(dim=1, unbiased=False)
    qts = [torch.quantile(flat, q/100.0, dim=1) for q in qs]
    return mins, maxs, means, stds, qts

@torch.no_grad()
def print_input_stats(l1c_norm, l2a_norm, mean, std, step_label: str, per_band: bool = True, qs=(1,50,99)):
    l1c_ref = l1c_norm * std + mean
    l2a_ref = l2a_norm * std + mean
    for name, x in [("L1C", l1c_ref), ("L2A", l2a_ref)]:
        mins, maxs, means, stds, qts = bandwise_stats(x, qs=qs)
        frac_lt0 = (x < 0).float().mean().item()
        frac_gt1 = (x > 1).float().mean().item()
        print(f"[{step_label}] {name} reflectance: any<0={frac_lt0:.4f}, any>1={frac_gt1:.4f}")
        if per_band:
            header = f" band |    min     p{qs[0]:02d}     p{qs[1]:02d}     p{qs[2]:02d}     max     mean     std"
            print(header)
            for c in range(mins.numel()):
                print(f"{c:5d} | {mins[c]: .4f}   {qts[0][c]: .4f}   {qts[1][c]: .4f}   {qts[2][c]: .4f}   {maxs[c]: .4f}   {means[c]: .4f}   {stds[c]: .4f}")

@torch.no_grad()
def print_embedding_stats(encoder, l1c_norm, l2a_norm, step_label: str, pool: str):
    z1 = encoder.get_patch_embeddings(l1c_norm)
    z2 = encoder.get_patch_embeddings(l2a_norm)
    n1 = z1.norm(dim=-1); n2 = z2.norm(dim=-1)
    z1n = F.normalize(z1, dim=-1); z2n = F.normalize(z2, dim=-1)
    cos = (z1n * z2n).sum(dim=-1)
    def s(x): return x.mean().item(), x.std().item(), x.min().item(), x.max().item()
    m1,s1,a1,b1 = s(n1); m2,s2,a2,b2 = s(n2); mc,sc,ac,bc = s(cos)
    print(f"[{step_label}] patch-embed norms L1C mean={m1:.3f}±{s1:.3f} range[{a1:.3f},{b1:.3f}] | L2A mean={m2:.3f}±{s2:.3f} range[{a2:.3f},{b2:.3f}]")
    print(f"[{step_label}] L1C vs L2A cosine (per token): mean={mc:.4f}±{sc:.4f} range[{ac:.4f},{bc:.4f}]")
    if pool == "mean":
        z1m = z1.mean(dim=1); z2m = z2.mean(dim=1)
        cosm = F.cosine_similarity(z1m, z2m, dim=-1).mean().item()
        print(f"[{step_label}] pooled-mean cosine: {cosm:.4f}")

def compute_loss(pred, tgt, cfg, adapter):
    loss = cosine_loss(pred, tgt)
    lam_l2 = float(cfg["SOLVER"].get("lambda_l2", 0.0))
    lam_coral = float(cfg["SOLVER"].get("lambda_coral", 0.0))
    lam_id = float(cfg["SOLVER"].get("lambda_id", 0.0))
    if lam_l2 > 0:
        loss = loss + lam_l2 * l2_loss(pred, tgt)
    if lam_coral > 0:
        a = pred.reshape(-1, pred.shape[-1]); b = tgt.reshape(-1, tgt.shape[-1])
        loss = loss + lam_coral * coral_loss(a, b)
    if lam_id > 0 and hasattr(adapter, "identity_reg"):
        loss = loss + lam_id * adapter.identity_reg()
    return loss

def train_epoch(encoder, adapter, loader, device, cfg, optimizer, scaler, mean, std, epoch: int, pool: str, adapter_space: str):
    encoder.eval(); adapter.train()
    total, steps = 0.0, 0
    sanity_batches = int(cfg.get("LOGGING",{}).get("sanity_batches", 2))
    per_band = bool(cfg.get("LOGGING",{}).get("per_band", True))
    qs = tuple(cfg.get("LOGGING",{}).get("percentiles", [1,50,99]))

    pbar = tqdm(loader, total=None, desc=f"train[{epoch:03d}]", dynamic_ncols=True, leave=False)
    for l1c, l2a in pbar:
        l1c = l1c.to(device, non_blocking=True)
        l2a = l2a.to(device, non_blocking=True)
        l1c = normalize_tensor(l1c, mean, std)
        l2a = normalize_tensor(l2a, mean, std)

        l1c_safe = l1c.detach().contiguous().clone()
        l2a_safe = l2a.detach().contiguous().clone()

        if steps < sanity_batches:
            print_input_stats(l1c_safe, l2a_safe, mean, std, step_label=f"train[b{steps}]", per_band=per_band, qs=qs)
            print_embedding_stats(encoder, l1c_safe, l2a_safe, step_label=f"train[b{steps}]", pool=pool)

        if steps < int(cfg.get("LOGGING",{}).get("sanity_batches", 2)):
            with torch.no_grad():
                z1 = encoder.get_patch_embeddings(l1c_safe)
                z2 = encoder.get_patch_embeddings(l2a_safe)
                mae = (z1 - z2).abs().mean().item()
                mxx = (z1 - z2).abs().max().item()
                print(f"[{('train' if encoder.training else 'valid')}[b{steps}]] embed diff MAE={mae:.6e} MAX={mxx:.6e}")


        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and scaler.is_enabled()

        if use_amp:
            with torch.cuda.amp.autocast():
                tgt = get_embeddings(encoder, l2a_safe, pool=pool, need_grad=False)
                if adapter_space == "embedding":
                    src = get_embeddings(encoder, l1c_safe, pool=pool, need_grad=False)
                    if hasattr(adapter, "update_stats"):
                        adapter.update_stats(src.detach(), tgt.detach())
                    pred = adapter(src)
                else:
                    x_hat = adapter(l1c_safe)
                    pred = get_embeddings(encoder, x_hat, pool=pool, need_grad=True)
                loss = compute_loss(pred, tgt, cfg, adapter)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
        else:
            tgt = get_embeddings(encoder, l2a_safe, pool=pool, need_grad=False)
            if adapter_space == "embedding":
                src = get_embeddings(encoder, l1c_safe, pool=pool, need_grad=False)
                if hasattr(adapter, "update_stats"):
                    adapter.update_stats(src.detach(), tgt.detach())
                pred = adapter(src)
            else:
                x_hat = adapter(l1c_safe)
                pred = get_embeddings(encoder, x_hat, pool=pool, need_grad=True)
            loss = compute_loss(pred, tgt, cfg, adapter)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

        val = float(loss.item())
        total += val; steps += 1
        pbar.set_postfix(loss=f"{val:.6f}")
    pbar.close()
    return total / max(1, steps)

@torch.no_grad()
def eval_epoch(encoder, adapter, loader, device, cfg, mean, std, epoch: int, pool: str, adapter_space: str):
    encoder.eval(); adapter.eval()
    total, steps = 0.0, 0
    sanity_batches = int(cfg.get("LOGGING",{}).get("sanity_batches", 2))
    per_band = bool(cfg.get("LOGGING",{}).get("per_band", True))
    qs = tuple(cfg.get("LOGGING",{}).get("percentiles", [1,50,99]))

    pbar = tqdm(loader, total=None, desc=f"valid[{epoch:03d}]", dynamic_ncols=True, leave=False)
    for l1c, l2a in pbar:
        l1c = l1c.to(device, non_blocking=True)
        l2a = l2a.to(device, non_blocking=True)
        l1c = normalize_tensor(l1c, mean, std)
        l2a = normalize_tensor(l2a, mean, std)

        l1c_safe = l1c.detach().contiguous().clone()
        l2a_safe = l2a.detach().contiguous().clone()

        if steps < sanity_batches:
            print_input_stats(l1c_safe, l2a_safe, mean, std, step_label=f"valid[b{steps}]", per_band=per_band, qs=qs)
            print_embedding_stats(encoder, l1c_safe, l2a_safe, step_label=f"valid[b{steps}]", pool=pool)

        tgt = get_embeddings(encoder, l2a_safe, pool=pool, need_grad=False)
        if adapter_space == "embedding":
            src = get_embeddings(encoder, l1c_safe, pool=pool, need_grad=False)
            pred = adapter(src)
        else:
            x_hat = adapter(l1c_safe)
            pred = get_embeddings(encoder, x_hat, pool=pool, need_grad=False)
        loss = compute_loss(pred, tgt, cfg, adapter)

        val = float(loss.item())
        total += val; steps += 1
        pbar.set_postfix(loss=f"{val:.6f}")
    pbar.close()
    return total / max(1, steps)

@hydra.main(version_base=None, config_path=str(CONFIG_PATH))
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_cfg = cfg["MODEL"]; data_cfg = cfg["DATASETS"]; solver = cfg["SOLVER"]; ckpt_cfg = cfg["CHECKPOINT"]

    print("Loading MS-CLIP...")
    msclip_model, _, _ = build_msclip(
        model_name=model_cfg["msclip_model_name"],
        pretrained=True,
        ckpt_path=model_cfg.get("msclip_ckpt", ""),
        device=device,
        channels=int(model_cfg["channels"]),
    )
    encoder = msclip_model.image_encoder

    move_to_device(msclip_model, device)
    encoder = msclip_model.image_encoder

    encoder.eval()

    dummy = torch.zeros(1, int(model_cfg["channels"]), int(model_cfg["img_res"]), int(model_cfg["img_res"]), device=device)
    with torch.no_grad():
        embed_dim = int(encoder.get_patch_embeddings(dummy).shape[-1])
    print(f"Embedding dim: {embed_dim}")

    # --- Sanity probe: ensure encoder output changes when input changes ---
    enc_dev = next(encoder.parameters()).device
    probe_shape = (1, int(model_cfg["channels"]), int(model_cfg["img_res"]), int(model_cfg["img_res"]))
    x1 = torch.rand(*probe_shape, device=enc_dev)
    x2 = torch.rand(*probe_shape, device=enc_dev)

    # Try to disable any internal preprocessing/caching if present
    for attr in ("normalize_inputs", "cache_outputs", "use_cache", "training_cache"):
        if hasattr(encoder, attr):
            try:
                setattr(encoder, attr, False)
                print(f"[probe] set encoder.{attr}=False")
            except Exception:
                pass

    with torch.no_grad():
        z1 = encoder.get_patch_embeddings(x1.clone().contiguous())
        z2 = encoder.get_patch_embeddings(x2.clone().contiguous())

    diff = (z1 - z2).abs().mean().item()
    cos  = torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(z1.flatten(1), dim=1),
        torch.nn.functional.normalize(z2.flatten(1), dim=1),
        dim=1
    ).mean().item()
    print(f"[probe] encoder diff MAE={diff:.6e}  cosine={cos:.6f}")
    if diff == 0.0 or cos > 0.9999:
        raise RuntimeError(
            "Encoder returns (nearly) identical embeddings for different random inputs. "
            "This indicates internal caching or a preprocessing bug. See patch B below."
        )


    adapter, adapter_space = build_adapter(
        model_type=model_cfg.get("model_type", "embed_affine"),
        embed_dim=embed_dim,
        in_channels=int(model_cfg["channels"]),
        rank=int(model_cfg.get("rank", 16)),
        wct_momentum=float(model_cfg.get("wct_momentum", 0.01)),
    )
    adapter = adapter.to(device)
    pool = str(model_cfg.get("pool", "token"))
    print(f"Adapter: {model_cfg.get('model_type','embed_affine')} | Space: {adapter_space} | Pool: {pool}")

    mean_file = cfg["DATASETS"]["kwargs"].get("mean_file", "")
    std_file  = cfg["DATASETS"]["kwargs"].get("std_file", "")

    bands = cfg["DATASETS"]["kwargs"].get("bands", "")

    means = extract_stats(mean_file, bands)[0,:,0,0]
    stds = extract_stats(std_file, bands)[0,:,0,0]

    assert len(means) == int(model_cfg["channels"]) and len(stds) == int(model_cfg["channels"]), "means/stds length must match MODEL.channels"
    mean_t, std_t = make_norm_tensors(means, stds, device=device)

    if bands is None:
        band_order = list(range(int(model_cfg["channels"])))
    else:
        band_order = list(range(len(bands)))
    print(f"Bands ({len(band_order)}): {bands if bands is not None else 'range'}")

    input_hw = (int(model_cfg.get("input_img_res", model_cfg["img_res"])), int(model_cfg.get("input_img_res", model_cfg["img_res"])))
    out_hw   = (int(model_cfg["img_res"]), int(model_cfg["img_res"]))

    train_dir = data_cfg["train"]["data_dir"]
    val_dir   = data_cfg["eval"]["data_dir"]
    io_bs     = int(data_cfg["eval"].get("io_batch_size", 64))

    train_ds = WorldStratPairsStream(train_dir, band_order=band_order, input_hw=input_hw, out_hw=out_hw, io_batch_size=io_bs)
    val_ds   = WorldStratPairsStream(val_dir,   band_order=band_order, input_hw=input_hw, out_hw=out_hw, io_batch_size=io_bs)

    def collate(batch):
        return (
            torch.stack([x[0] for x in batch], dim=0),
            torch.stack([x[1] for x in batch], dim=0),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=(int(data_cfg["train"]["num_workers"])>0),
        prefetch_factor=(2 if int(data_cfg["train"]["num_workers"])>0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["eval"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=(int(data_cfg["eval"]["num_workers"])>0),
        prefetch_factor=(2 if int(data_cfg["eval"]["num_workers"])>0 else None),
    )

    # after you build encoder, loaders, mean_t, std_t
    agg = compute_invariance_metrics(encoder, val_loader, mean_t, std_t, pool=model_cfg.get("pool","token"), max_batches=200)
    tex = latex_invariance_table(agg, title="MS-CLIP L1C↔L2A invariance on WorldStrat (val)")
    print("\n" + tex + "\n")

    # Optional: write to file
    out_dir = Path(cfg["CHECKPOINT"]["save_path"]) / cfg["CHECKPOINT"]["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "invariance_table.tex").write_text(tex)
    print(f"Wrote LaTeX table to {out_dir/'invariance_table.tex'}")

    bands = data_cfg["kwargs"].get("bands", None)

    opt = torch.optim.AdamW(adapter.parameters(), lr=float(solver["lr"]), weight_decay=float(solver.get("weight_decay", 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(solver.get("amp", False)))

    save_root = Path(ckpt_cfg["save_path"]) / ckpt_cfg["experiment_name"]
    save_root.mkdir(parents=True, exist_ok=True)
    best_path = save_root / f"adapter_{model_cfg.get('model_type','embed_affine')}_best.pt"

    epochs = int(solver.get("epochs", solver.get("num_epochs", 1)))
    best_val = float('inf')
    for epoch in range(1, epochs+1):
        tr = train_epoch(encoder, adapter, train_loader, device, cfg, opt, scaler, mean_t, std_t, epoch, pool, adapter_space)
        va = eval_epoch(encoder, adapter, val_loader, device, cfg, mean_t, std_t, epoch, pool, adapter_space)
        print(f"[{epoch:03d}/{epochs}] train_loss={tr:.6f}  val_loss={va:.6f}")
        if va < best_val:
            best_val = va
            torch.save(adapter.state_dict(), best_path)
            print(f"  ↳ New best (val_loss {best_val:.6f}). Saved: {best_path}")

    #final_path = save_root / f"adapter_{model_cfg.get('model_type','embed_affine')}_final.pt"
    #torch.save(adapter.state_dict(), final_path)
    #print(f"Saved final adapter to {final_path}")

if __name__ == "__main__":
    main()
