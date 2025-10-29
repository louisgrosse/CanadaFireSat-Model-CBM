# ============================================
# MS-CLIP Facto: pick positive samples + visualize
#  - temporal attention (from TemporalAttention)
#  - spatial occlusion heatmap (robust)
# Requirements: matplotlib, torch
# Inputs you must provide at the bottom:
#   - dataloader: yields {"inputs":[B,T,C,H,W], "labels":[B,1,H',W'] or [B,H',W'], "doy":[B,T] or None}
#   - model: an instance of MSClipFactorizeModel (or your LightningModule.model)
#   - ckpt_path: path to your .ckpt
# ============================================

import os, math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.data import get_data
import hydra
from omegaconf import DictConfig, OmegaConf
from src.constants import CONFIG_PATH
from DeepSatModels.utils.torch_utils import get_device
from pathlib import Path
from src.models import get_model
import sys
from tqdm import tqdm


# ---------- Utilities ----------

# ============================================
# MS-CLIP Facto: positives + RGB/Labels + Occlusion
# ============================================

import os, math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.data.utils import extract_stats

# ---------- Utilities ----------

def load_ckpt_into_model(model, ckpt_path, map_location="cpu", strict=False):
    sd = torch.load(ckpt_path, map_location=map_location)
    if "state_dict" in sd:
        sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=strict)
        except RuntimeError:
            fixed = {k.split("model.",1)[-1] if k.startswith("model.") else k: v for k,v in sd.items()}
            try:
                model.load_state_dict(fixed, strict=strict)
            except RuntimeError:
                if hasattr(model, "model"):
                    model.model.load_state_dict(fixed, strict=strict)
                else:
                    raise
    else:
        try:
            model.load_state_dict(sd, strict=strict)
        except RuntimeError:
            if hasattr(model, "model"):
                model.model.load_state_dict(sd, strict=strict)
            else:
                raise
    return model

def _get_labels(batch):
    for k in ("labels","label","y","mask","masks","target","targets"):
        if k in batch: return batch[k]
    raise KeyError("Batch has no labels key (expected one of labels/label/y/mask/masks/target/targets).")

@torch.no_grad()
def find_positive_samples(dataloader, device, max_samples=4, min_pos_frac=0.001):
    pos_batches = []
    for batch in tqdm(dataloader):
        x = batch["inputs"]
        y = _get_labels(batch)
        doy = batch.get("doy", None)

        if y.ndim == 3:
            y = (y.unsqueeze(1) > 0.5).float()
        elif y.ndim == 4:
            Hx = x.shape[-2] if x.ndim >= 4 else y.shape[2]
            Wx = x.shape[-1] if x.ndim >= 4 else y.shape[3]
            channel_last = (y.shape[-1] in (1,2,3,4) and y.shape[1] == Hx and y.shape[2] == Wx)
            if channel_last:
                y = y.permute(0, 3, 1, 2)
            if y.shape[1] == 1:
                y = (y > 0.5).float()
            else:
                y = y.permute(3, 0, 1, 2)
                y = (y.argmax(dim=1, keepdim=True) == 1).float()
        else:
            raise ValueError(f"Unexpected label shape: {tuple(y.shape)}")

        B = y.shape[0]
        pos_ratio = (y > 0.5).float().view(B, -1).mean(dim=1)
        keep = (pos_ratio >= min_pos_frac).nonzero(as_tuple=True)[0]
        for idx in keep.tolist():
            xb = x[idx:idx+1].to(device, non_blocking=True)
            yb = y[idx:idx+1].to(device, non_blocking=True)
            doyb = None if doy is None else doy[idx:idx+1].to(device, non_blocking=True)
            pos_batches.append({"inputs": xb, "labels": yb, "doy": doyb})
            if len(pos_batches) >= max_samples:
                return pos_batches
    return pos_batches


def predict_score(model, x, doy=None, cls=1):
    model.eval()
    out = forward_robust(model, x, doy)  # [1,K,H,W] or [1,1,H,W]
    if out.shape[1] == 1:
        score_map = torch.sigmoid(out[:,0])
    else:
        score_map = out[:, cls]
    return score_map.mean().item()

def forward_robust(model, x, doy=None):
    try:
        return model(x, doy)
    except TypeError:
        try:
            return model(x)
        except Exception:
            if hasattr(model, "model"):
                try:
                    return model.model(x, doy)
                except TypeError:
                    return model.model(x)
            raise RuntimeError("Could not call forward with any supported signature.")

def occlusion_heatmap(model, x, doy=None, cls=1, patch=32, stride=16, fill=0.0):
    model.eval()
    _, T, C, H, W = x.shape
    base = predict_score(model, x, doy, cls=cls)
    grid_h = max(1, (H - patch)//stride + 1)
    grid_w = max(1, (W - patch)//stride + 1)
    heat = torch.zeros((grid_h, grid_w), device=x.device)

    for i, y0 in enumerate(range(0, max(1, H - patch + 1), stride)):
        for j, x0 in enumerate(range(0, max(1, W - patch + 1), stride)):
            xb = x.clone()
            y1, x1 = min(H, y0+patch), min(W, x0+patch)
            xb[:, :, :, y0:y1, x0:x1] = fill
            s = predict_score(model, xb, doy, cls=cls)
            drop = max(0.0, base - s)
            heat[i, j] = drop

    heat = heat / (heat.max() + 1e-8)
    heat_up = F.interpolate(heat[None,None], size=(H, W), mode="bilinear", align_corners=False)[0,0]
    return heat_up

# ---------- Temporal attention capture ----------

class TempAttnCatcher:
    def __init__(self, model): self.model, self.hook, self.attn = model, None, None
    def __enter__(self):
        enc = getattr(self.model, "temp_enc", None)
        if enc is None and hasattr(self.model, "model"):
            enc = getattr(self.model.model, "temp_enc", None)
        assert enc is not None and hasattr(enc, "attn"), "temp_enc.attn not found on model."
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple) and len(output) == 2:
                self.attn = output[1].detach()  # [B*P, T, T]
        self.hook = enc.attn.register_forward_hook(hook_fn)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.hook is not None: self.hook.remove()

# ---------- Visualization helpers ----------

def _to_rgb(img_tc_hw, rgb_bands=(2,1,0), mean=None, std=None, t_sel=4, gamma=None):
    x = img_tc_hw
    if x.ndim == 4:
        t_sel = int(max(0, min(t_sel, x.shape[0]-1)))
        x = x[t_sel]  # [C,H,W]
    C, H, W = x.shape

    if mean is not None and std is not None:
        m = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(-1,1,1)[:C]
        s = torch.as_tensor(std,  device=x.device, dtype=x.dtype).view(-1,1,1)[:C]
        x = x * s + m

    r_i, g_i, b_i = [min(C-1, i) for i in rgb_bands]
    r, g, b = x[r_i], x[g_i], x[b_i]

    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    stacked = torch.stack([r.flatten(), g.flatten(), b.flatten()], dim=0)
    lo = torch.quantile(stacked, 0.02)
    hi = torch.quantile(stacked, 0.98)
    den = (hi - lo).clamp_min(1e-6)

    r = ((r - lo) / den).clamp(0, 1)
    g = ((g - lo) / den).clamp(0, 1)
    b = ((b - lo) / den).clamp(0, 1)

    if gamma is not None:
        r = r.pow(gamma); g = g.pow(gamma); b = b.pow(gamma)

    rgb = torch.stack([r, g, b], dim=-1)
    return (rgb * 255).byte().cpu().numpy()


def _make_label_vis(label_1hw):
    """label_1hw: [1,H,W] or [H,W] -> uint8 [H,W] 0/255 for plotting."""
    if label_1hw.ndim == 3:
        label = label_1hw[0]
    else:
        label = label_1hw
    label = (label > 0.5).to(torch.uint8) * 255
    return label.cpu().numpy()

def _overlay_heat_on_rgb(rgb_uint8, heat_01, alpha=0.5):
    """
    rgb_uint8: [H,W,3] uint8
    heat_01:   [H,W] float in [0,1]
    returns uint8 [H,W,3]
    """
    H, W, _ = rgb_uint8.shape
    heat = heat_01.detach().cpu().clamp(0,1)
    heat3 = heat.unsqueeze(-1).repeat(1,1,3)     # gray heat overlay (keep simple & dependency-free)
    base = torch.from_numpy(rgb_uint8).float()/255.0
    out = (1-alpha)*base + alpha*heat3
    return (out.clamp(0,1)*255).byte().numpy()

# ---------- Driver ----------

def visualize_on_positive_examples(
    dataloader,
    model,
    ckpt_path,
    device="cuda",
    num_samples=3,
    min_pos_frac=0.0005,
    target_class=1,
    patch=32,
    stride=16,
    rgb_bands=(2,1,0),
    out_dir="heatmap_viz",     
    prefix="sample",           
    save_temporal=True,         
    dpi=150,                    
    mean_file=None,
    std_file=None,
    cfg=None,
):
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
    model = load_ckpt_into_model(model, ckpt_path, map_location=device, strict=False)
    model.to(device).eval()

    # Resolve DataLoader if a DataModule was passed
    try:
        iter(dataloader)
    except TypeError:
        if hasattr(dataloader, "val_dataloader"):
            dataloader = dataloader.val_dataloader()
        elif hasattr(dataloader, "test_dataloader"):
            dataloader = dataloader.test_dataloader()
        else:
            dataloader = dataloader.train_dataloader()

    pos_batches = find_positive_samples(dataloader, device, max_samples=num_samples, min_pos_frac=min_pos_frac)
    if not pos_batches:
        print("[warn] No positive samples meeting the criterion—try lowering `min_pos_frac` or check labels.")
        return []
    
    bands = cfg["DATASETS"]["kwargs"].get("bands", "")
    
    means = extract_stats(mean_file, bands)[0,:,0,0]
    stds = extract_stats(std_file, bands)[0,:,0,0]

    saved_paths = []

    for si, batch in enumerate(pos_batches, 1):
        x = batch["inputs"]   # [1,T,C,H,W]
        y = batch["labels"]   # [1,1,H,W] or [1,H,W]
        doy = batch.get("doy", None)

        # Temporal attention capture (optional)
        with TempAttnCatcher(model) as catcher:
            _ = forward_robust(model, x, doy)
        A = catcher.attn  # [B*P, T, T] or None

        # Spatial occlusion heatmap
        heat = occlusion_heatmap(model, x, doy=doy, cls=target_class, patch=patch, stride=stride)  # [H,W] in [0,1]

        # Prepare visuals
        _, T, C, H, W = x.shape
        print(T, C, H, W)


        rgb = _to_rgb(x[0], rgb_bands=rgb_bands,mean=means,std=stds)               # [H,W,3] uint8
        lab = _make_label_vis(y[0])                            # [H,W] uint8
        overlay = _overlay_heat_on_rgb(rgb, heat, alpha=0.5)   # [H,W,3] uint8

        # ---------- Save 4-panel composite ----------
        fig = plt.figure(figsize=(16,4))
        ax1 = plt.subplot(1,4,1); ax1.set_title(f"RGB (t=0) bands={rgb_bands}"); ax1.imshow(rgb); ax1.axis("off")
        ax2 = plt.subplot(1,4,2); ax2.set_title("Label mask"); ax2.imshow(lab, cmap="gray"); ax2.axis("off")
        ax3 = plt.subplot(1,4,3); ax3.set_title("Occlusion heatmap")
        im3 = ax3.imshow(heat.detach().cpu(), interpolation="nearest"); ax3.axis("off")
        cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax4 = plt.subplot(1,4,4); ax4.set_title("Overlay (heat on RGB)"); ax4.imshow(overlay); ax4.axis("off")
        plt.suptitle(f"Sample {si}")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{prefix}_{si:02d}_composite.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

        # ---------- (Optional) save temporal summary ----------
        if save_temporal and (A is not None):
            A_time = A.mean(dim=1)  # [B*P, T]
            avg_curve = A_time.mean(dim=0).cpu().numpy()  # [T]

            fig_t = plt.figure(figsize=(10,3))
            plt.title(f"Sample {si}: Temporal attention (avg over patches)")
            plt.plot(avg_curve)
            plt.xlabel("time step")
            plt.ylabel("importance (a.u.)")
            plt.grid(True)
            out_t_path = os.path.join(out_dir, f"{prefix}_{si:02d}_temporal.png")
            fig_t.savefig(out_t_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_t)
            saved_paths.append(out_t_path)

    print(f"[info] Saved {len(saved_paths)} files to: {os.path.abspath(out_dir)}")
    return saved_paths

@hydra.main(version_base=None, config_path="/home/grosse/CanadaFireSat-Model-CBM/results/models/MS-CLIP_Fixed_NoNoise/")
def generate_heat_maps(cfg: DictConfig):
    """Training and Evaluation (Val) Endpoint"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Extract key variables from the config
    num_epochs = cfg["SOLVER"]["num_epochs"]
    save_steps = cfg["CHECKPOINT"]["save_steps"]
    save_path = cfg["CHECKPOINT"]["save_path"]
    save_path = Path(save_path) / cfg["CHECKPOINT"]["experiment_name"]
    cfg["CHECKPOINT"]["save_path"] = str(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg["CHECKPOINT"]["load_from_checkpoint"]
    local_device_ids = cfg["SET-UP"]["local_device_ids"]  # List of integers ids for GPUs
    arch_name = cfg["MODEL"]["architecture"]
    device = get_device(local_device_ids, allow_cpu=False)
    
    mean_file = cfg["DATASETS"]["kwargs"].get("mean_file", "")
    std_file  = cfg["DATASETS"]["kwargs"].get("std_file", "")

    bands = cfg["DATASETS"]["kwargs"].get("bands", "")

    datamodule = get_data(cfg)
    cfg["SOLVER"]["num_steps_train"] = len(datamodule.train_dataloader())

    datamodule = get_data(cfg)

    model = get_model(cfg,device)

    saved = visualize_on_positive_examples(
        datamodule.train_dataloader(),
        model,
        "./results/models/MS-CLIP_Fixed/MSClipFacto-27-step-40000.00.ckpt",
        device="cuda",
        num_samples=3,
        min_pos_frac=0.001,
        target_class=1,
        patch=32,
        stride=16,
        rgb_bands=(2,1,0),
        prefix="val",
        save_temporal=True,
        dpi=150,
        out_dir="./results/models/visualisations/",
        mean_file = mean_file,
        std_file = std_file,
        cfg=cfg,
    )

    print(saved)


if __name__ == "__main__":
    generate_heat_maps()

