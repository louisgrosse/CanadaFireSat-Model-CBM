"""Evaluation Script for Validation, Test, and Test Hard"""
from pathlib import Path

# --- HPC-safe, headless plotting ---
import matplotlib
matplotlib.use("Agg")  # headless backend for clusters
import matplotlib.pyplot as plt

import os
import time
import random
import logging
from typing import Sequence, Tuple, Any, Optional, Iterable

import numpy as np
import torch
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

from DeepSatModels.metrics.numpy_metrics import get_classification_metrics
from src.constants import CONFIG_PATH
from src.data import get_data
from src.data.utils import segmentation_ground_truths
from src.eval.utils import get_pr_auc_scores
from src.models.module_img import ImgModule
from src.models.module_tab import TabModule

# ---------- logging config ----------
logging.basicConfig(level=logging.INFO)  # set to DEBUG for per-slot details
log = logging.getLogger(__name__)
# ------------------------------------


# ---------------------- Metrics logging helpers ----------------------
def _summ(v: Any) -> str:
    try:
        if isinstance(v, torch.Tensor):
            base = f"Tensor{tuple(v.shape)} dtype={v.dtype} device={v.device}"
            prev = v.flatten()[:3].detach().cpu().numpy()
            return f"{base} preview={np.array2string(prev, precision=4)}"
        if isinstance(v, np.ndarray):
            base = f"ndarray{v.shape} dtype={v.dtype}"
            prev = v.flatten()[:3]
            return f"{base} preview={np.array2string(prev, precision=4)}"
        if isinstance(v, (list, tuple)):
            base = f"{type(v).__name__}[len={len(v)}]"
            prev = v[:3]
            return f"{base} preview={prev}"
        return f"{type(v).__name__} value={str(v)[:60]}"
    except Exception as e:
        return f"{type(v).__name__} (summary-error: {e})"


def _coerce_seq(name: str, obj: Any) -> Sequence:
    log.debug(f"[metrics] raw {name}: type={type(obj).__name__}, summary={_summ(obj)}")
    if isinstance(obj, (list, tuple)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 1:
            return obj.detach().cpu().tolist()
        return [t.detach().cpu().numpy() for t in obj]
    log.warning(f"[metrics] {name} is not a sequence; wrapping into a list.")
    return [obj]


def _expect_tuple(name: str, seq: Sequence, expected_len: int) -> Tuple:
    L = len(seq)
    if L > expected_len:
        extras = list(seq[expected_len:])
        log.warning(
            f"[metrics] {name} has {L} items, expected {expected_len}. "
            f"Truncating. Extras (preview): {[ _summ(e) for e in extras[:3] ]}"
        )
        seq = seq[:expected_len]
    elif L < expected_len:
        log.error(f"[metrics] {name} has {L} items, expected {expected_len}. Padding with None.")
        seq = list(seq) + [None] * (expected_len - L)
    for i, v in enumerate(seq):
        log.debug(f"[metrics] {name}[{i}]: {_summ(v)}")
    return tuple(seq)


def _safe_idx(arr, idx: int, default=np.nan):
    try:
        if arr is None:
            return default
        if isinstance(arr, torch.Tensor):
            return float(arr.detach().cpu().flatten()[idx])
        if isinstance(arr, np.ndarray):
            return float(arr.flatten()[idx])
        return float(arr[idx])
    except Exception:
        return default
# --------------------------------------------------------------------


# ---------------------- Visualization helpers ----------------------
def get_rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("SLURM_PROCID", "0"))


def _to_chw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        if x.shape[0] <= x.shape[-1] and x.shape[0] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            return x.permute(2, 0, 1).contiguous()
        return x
    while x.ndim > 4:
        x = x[0]
    if x.ndim == 4:
        b, d1, d2, d3 = x.shape
        if d3 in range(1, 33) and d1 >= 32 and d2 >= 32:
            return x[0].permute(2, 0, 1).contiguous()
        else:
            return x[0]
    raise ValueError(f"Expected 3–5D tensor, got {x.ndim}D with shape {tuple(x.shape)}")


def _pca_to_rgb(chw: torch.Tensor) -> torch.Tensor:
    C, H, W = chw.shape
    x = chw.detach().float().reshape(C, -1)
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    xs = (x - mean) / std
    cov = (xs @ xs.t()) / (xs.shape[1] - 1)
    U, S, V = torch.linalg.svd(cov)
    Wp = U[:, :3]
    rgb = (Wp.t() @ xs).reshape(3, H, W)
    rgb_min = rgb.amin(dim=(1, 2), keepdim=True)
    rgb_max = rgb.amax(dim=(1, 2), keepdim=True)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)
    rgb = (rgb * 255.0).clamp(0, 255)
    return rgb


def _denorm_inplace(x_sel: torch.Tensor, mean: Iterable[float], std: Iterable[float], idxs: Iterable[int]) -> None:
    if mean is None or std is None:
        return
    if not isinstance(x_sel, torch.Tensor):
        return
    try:
        for i, band_idx in enumerate(idxs):
            m = float(mean[band_idx])
            s = float(std[band_idx])
            x_sel[i] = x_sel[i] * s + m
    except Exception as e:
        log.warning(f"[vis] Denorm failed (using normalized values). Reason: {e}")


def _rgb_from_ms_tensor(
    x: torch.Tensor,
    band_map: Optional[tuple],
    denorm_mean: Optional[Iterable[float]],
    denorm_std: Optional[Iterable[float]],
) -> torch.Tensor:
    """
    Returns a 3xHxW float tensor in [0,255] (not uint8), built from the
    multispectral tensor so we can rescale with bilinear safely.
    """
    x = _to_chw(x).detach().cpu().float()  # CxHxW
    if band_map is None:
        return _pca_to_rgb(x)
    assert len(band_map) == 3
    x_sel = x[list(band_map), :, :].clone()  # 3xHxW
    if denorm_mean is not None and denorm_std is not None:
        _denorm_inplace(x_sel, denorm_mean, denorm_std, band_map)
    # per-channel min-max -> [0,255]
    cmin = x_sel.amin(dim=(1, 2), keepdim=True)
    cmax = x_sel.amax(dim=(1, 2), keepdim=True)
    x_sel = (x_sel - cmin) / (cmax - cmin + 1e-6)
    x_sel = (x_sel * 255.0).clamp(0, 255)
    return x_sel  # 3xHxW (float)


def tensor_to_rgb(
    x: torch.Tensor,
    band_map: Optional[tuple] = (2, 1, 0),  # MS-CLIP order [B2,B3,B4,...] -> RGB=(B4,B3,B2)
    denorm_mean: Optional[Iterable[float]] = None,
    denorm_std: Optional[Iterable[float]] = None,
    fallback_to_pca: bool = True,
) -> "np.ndarray":
    if band_map is not None:
        x_sel = _rgb_from_ms_tensor(x, band_map, denorm_mean, denorm_std)  # 3xHxW [0,255]
        return x_sel.permute(1, 2, 0).byte().numpy()
    # PCA or trivial cases
    x = _to_chw(x).detach().cpu().float()
    C = x.shape[0]
    if C == 1:
        x = x.repeat(3, 1, 1)
        cmin = x.amin(dim=(1, 2), keepdim=True)
        cmax = x.amax(dim=(1, 2), keepdim=True)
        x = (x - cmin) / (cmax - cmin + 1e-6)
        x = (x * 255.0).clamp(0, 255)
        return x.permute(1, 2, 0).byte().numpy()
    if C == 3:
        cmin = x.amin(dim=(1, 2), keepdim=True)
        cmax = x.amax(dim=(1, 2), keepdim=True)
        x = (x - cmin) / (cmax - cmin + 1e-6)
        x = (x * 255.0).clamp(0, 255)
        return x.permute(1, 2, 0).byte().numpy()
    if fallback_to_pca:
        rgb = _pca_to_rgb(x)
        return rgb.permute(1, 2, 0).byte().numpy()
    y = x[:3]
    cmin = y.amin(dim=(1, 2), keepdim=True)
    cmax = y.amax(dim=(1, 2), keepdim=True)
    y = (y - cmin) / (cmax - cmin + 1e-6)
    y = (y * 255.0).clamp(0, 255)
    return y.permute(1, 2, 0).byte().numpy()


def overlay_mask(img_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.4):
    overlay = img_np.copy()
    red = overlay[..., 0]
    red = np.clip(red + alpha * (mask_np.astype(np.float32)), 0, 1)
    overlay[..., 0] = red
    return overlay


def save_triptych(img_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray,
                  save_path: Path, figsize=(10, 3), dpi=150):
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    axes[0].imshow(img_np); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(gt_np, vmin=0, vmax=1); axes[1].set_title("GT (1=fire)"); axes[1].axis("off")
    axes[2].imshow(pred_np, vmin=0, vmax=1); axes[2].set_title("Pred (1=fire)"); axes[2].axis("off")
    fig.tight_layout(); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)


def save_side_by_side(img_np: np.ndarray, gt_np: np.ndarray, save_path: Path,
                      figsize=(8, 4), dpi=150, titles=("Image", "GT")):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axes[0].imshow(img_np); axes[0].set_title(titles[0]); axes[0].axis("off")
    axes[1].imshow(gt_np, vmin=0, vmax=1); axes[1].set_title(titles[1]); axes[1].axis("off")
    fig.tight_layout(); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
# -------------------------------------------------------------------


# ---------------------- Output path helpers ----------------------
def _is_writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        with open(test, "w") as _f: _f.write("ok")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False
# ----------------------------------------------------------------


def _downscale_rgb_to(x_rgb_3chw: torch.Tensor, H: int, W: int) -> np.ndarray:
    """
    Downscale a 3xHxW float tensor [0,255] to (H,W) using bilinear for display.
    Returns HxWx3 uint8.
    """
    t = x_rgb_3chw.unsqueeze(0)  # 1x3xH0xW0
    t_small = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0]
    t_small = t_small.clamp(0, 255).permute(1, 2, 0).byte().numpy()
    return t_small


def _to_hw_mask_np(mask_t: torch.Tensor) -> np.ndarray:
    """
    Make sure mask is (H, W) uint8 for matplotlib.
    Accepts (H,W), (1,H,W), (H,W,1) tensors of any dtype in {0,1,..}.
    """
    m = mask_t.detach().cpu()
    if m.ndim == 3:
        # squeeze channel if it's singleton in first or last pos
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            # unknown 3D layout; best effort: take first slice
            m = m[0]
    # now expect 2D
    m = m.long()
    return m.numpy().astype(np.uint8)


def _save_sanity_samples(dataset, out_dir: Path, n: int, band_map, denorm_mean, denorm_std, save_format="png"):
    """
    Dump n positive samples (image + GT only) with NO mask upsampling.
    - If sample has 'labels_raw', use it.
    - Else use 'labels' as-is.
    In both cases, the **image** is downscaled to the mask resolution for clean alignment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    positives_seen = 0

    for i in range(len(dataset)):
        data = dataset[i]
        sample = data[0]

        # current inputs (post transforms)
        x = sample["inputs"]  # CHW
        # get ground truth tensors
        labels, _ = segmentation_ground_truths(sample)  # possibly downsampled
        labels_raw = None
        for k in ("labels_raw", "labels_orig", "labels_pre_down", "mask_raw"):
            if k in sample and isinstance(sample[k], torch.Tensor):
                labels_raw = sample[k]
                break

        gt_t = labels_raw if labels_raw is not None else labels

        # Only keep positives
        if (gt_t == 1).any():
            positives_seen += 1

            # Build true-color rgb from MS tensor (float [0,255], 3xHxW)
            rgb_3chw = _rgb_from_ms_tensor(x, band_map, denorm_mean, denorm_std)

            # Downscale image to GT size (NO mask upsampling)
            gt_np = _to_hw_mask_np(gt_t)
            Hm, Wm = int(gt_np.shape[-2]), int(gt_np.shape[-1])
            img_small = _downscale_rgb_to(rgb_3chw, Hm, Wm)   # Hm x Wm x 3 (uint8)

            fname = f"sanity_{i:06d}.{save_format}"
            save_side_by_side(img_small, gt_np, out_dir / fname, figsize=(8, 4), dpi=150)

            saved += 1
            if saved >= n:
                break

    log.info(f"[sanity] positives_seen={positives_seen}, saved={saved} samples at {out_dir}")


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval")
@torch.no_grad()
def evaluate(cfg: DictConfig):
    """Evaluation Endpoint"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # -------- Visualization config --------
    vis_cfg = cfg.get("VIS", {})
    VIS_ENABLE: bool = bool(vis_cfg.get("enable", True))
    VIS_DIR_USER = vis_cfg.get("vis_dir", None)
    NUM_VIS: int = int(vis_cfg.get("num_vis", 50))
    P_VIS_POS: float = float(vis_cfg.get("p_vis_pos", 0.0))
    VIS_OVERLAY: bool = bool(vis_cfg.get("overlay", False))
    RANK_SUBDIRS: bool = bool(vis_cfg.get("rank_subdirs", True))
    FIGSIZE = tuple(vis_cfg.get("figsize", (10, 3)))
    DPI = int(vis_cfg.get("dpi", 150))
    SAVE_FORMAT = str(vis_cfg.get("save_format", "png")).lower()
    assert SAVE_FORMAT in {"png", "webp"}, "VIS.save_format must be 'png' or 'webp'"

    # NEW: sanity dump
    SANITY_N: int = int(vis_cfg.get("sanity_check_n", 0))

    # NEW: MS-CLIP band map & denorm
    VIS_BAND_MAP = tuple(vis_cfg.get("band_map", (2, 1, 0)))  # B4,B3,B2 in [B2,B3,B4,...]
    VIS_DENORM_MEAN = vis_cfg.get("denorm_mean", None)
    VIS_DENORM_STD  = vis_cfg.get("denorm_std", None)

    RANK = get_rank()
    vis_saved = 0
    positives_seen = 0

    # Device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])

    # Load model (unchanged) ...
    if cfg["mode"] == "image":
        try:
            model = ImgModule.load_from_checkpoint(cfg["model_path"])
        except (KeyError, RuntimeError):
            with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            model = ImgModule(model_cfg)
            mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
            print("Missing keys:", mis_keys)

        if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
            model.model.out_H = cfg["MODEL"]["out_H"]
            model.model.out_W = cfg["MODEL"]["out_W"]

        if "ViT" in model.model_type:
            if model.model.features.patch_embed.img_size != (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"]):
                model.model.features.patch_embed.img_size = (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"])
    else:
        try:
            model = TabModule.load_from_checkpoint(cfg["model_path"])
        except KeyError or RuntimeError:
            with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            model = TabModule(model_cfg)
            mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
            print("Missing keys:", mis_keys)
            print("Unexpected keys:", un_keys)

        if model.model_type in ["TabTSViT", "TabConvLSTM"]:
            if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
                model.model.out_H = cfg["MODEL"]["out_H"]
                model.model.out_W = cfg["MODEL"]["out_W"]
        elif model.model_type in ["EnvResNet", "EnvViTFactorizeModel"]:
            if (cfg["MODEL"]["out_H"] != model.model.mid_model.out_H
                or cfg["MODEL"]["out_W"] != model.model.mid_model.out_W):
                model.model.mid_model.out_H = cfg["MODEL"]["out_H"]
                model.model.mid_model.out_W = cfg["MODEL"]["out_W"]
        else:
            if (cfg["MODEL"]["out_H"] != model.model.sat_model.out_H
                or cfg["MODEL"]["out_W"] != model.model.sat_model.out_W):
                model.model.sat_model.out_H = cfg["MODEL"]["out_H"]
                model.model.sat_model.out_W = cfg["MODEL"]["out_W"]

        if "ViT" in model.model_type and model.model_type != "EnvViTFactorizeModel":
            if model.model.sat_model.features.patch_embed.img_size != (
                cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"]):
                model.model.sat_model.features.patch_embed.img_size = (
                    cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"])

    model.eval(); model.to(device)

    # Data
    datamodule = get_data(cfg)
    dataset = datamodule.test_dataloader(split=cfg["split"]).dataset

    # Output dirs (unchanged)
    if "test_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["test_max_seq_len"])
    elif "env_val_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["env_val_max_seq_len"])
    else:
        temp_size = "adapt"

    temp_size = (temp_size + f"_{cfg['DATASETS']['kwargs']['eval_sampling']}"
                 if "eval_sampling" in cfg["DATASETS"]["kwargs"] else temp_size)
    spa_size = str(cfg["MODEL"]["img_res"]) if "img_res" in cfg["MODEL"] else str(cfg["MODEL"]["mid_input_res"])

    if cfg["DATASETS"]["eval"].get("hard"):
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}_hard"
    else:
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"
    output_dir = output_dir.resolve()

    if VIS_DIR_USER is not None and str(VIS_DIR_USER).strip():
        base_vis_dir = Path(VIS_DIR_USER).resolve()
    else:
        base_vis_dir = (output_dir / "vis").resolve()
    vis_dir = base_vis_dir / f"rank{RANK}" if RANK_SUBDIRS else base_vis_dir

    # separate sanity dir
    base_sanity_dir = (output_dir / "vis_sanity").resolve()
    sanity_dir = base_sanity_dir / f"rank{RANK}" if RANK_SUBDIRS else base_sanity_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if VIS_ENABLE:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if SANITY_N > 0:
        sanity_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[eval] CWD: {Path.cwd().resolve()}")
    log.info(f"[eval] Output dir: {output_dir}")
    log.info(f"[eval] Vis dir: {vis_dir} (enabled={VIS_ENABLE})")
    if SANITY_N > 0:
        log.info(f"[sanity] Sanity dir: {sanity_dir} (N={SANITY_N})")

    # --- Sanity: dump positives with NO mask upsampling ---
    if SANITY_N > 0:
        _save_sanity_samples(
            dataset=dataset,
            out_dir=sanity_dir,
            n=SANITY_N,
            band_map=VIS_BAND_MAP,
            denorm_mean=VIS_DENORM_MEAN,
            denorm_std=VIS_DENORM_STD,
            save_format=SAVE_FORMAT,
        )

    # ------- Evaluation loop (unchanged metrics) -------
    tot_preds, tot_labels, tot_losses, tot_probs = [], [], [], []
    tot_regions, tot_fwi = [], []

    for i in tqdm(range(len(dataset)), desc=f"Inferring on split {cfg['split']}", total=len(dataset)):
        data = dataset[i]
        with torch.no_grad():
            if cfg["mode"] == "image":
                sample = data[0]; img_name_info = data[1]
                logits = model(sample["inputs"].unsqueeze(0).to(device))
            else:
                if model.model_type in [
                    "TabTSViT","TabConvLSTM","TabResNetConvLSTM",
                    "TabViTFactorizeModel","MultiViTFactorizeModel","MSClipFacto",
                ]:
                    sample = data[0]; tab_sample = data[1]; img_name_info = data[2]
                    logits = model(
                        sample["inputs"].unsqueeze(0).to(device),
                        tab_sample["tab_inputs"].unsqueeze(0).to(device),
                        tab_sample["mask"].unsqueeze(0).to(device),
                    )
                elif model.model_type in ["EnvResNet","EnvViTFactorizeModel"]:
                    sample = data[0]; img_name_info = data[1]
                    logits = model(
                        xmid=sample["mid_inputs"].unsqueeze(0).to(device),
                        xlow=sample["low_inputs"].unsqueeze(0).to(device),
                        m_mid=sample["mid_inputs_mask"].unsqueeze(0).to(device),
                        m_low=sample["low_inputs_mask"].unsqueeze(0).to(device),
                    )
                else:
                    sample = data[0]; env_sample = data[1]; img_name_info = data[2]
                    logits = model(
                        x=sample["inputs"].unsqueeze(0).to(device),
                        xmid=env_sample["mid_inputs"].unsqueeze(0).to(device),
                        xlow=env_sample["low_inputs"].unsqueeze(0).to(device),
                        m_mid=env_sample["mid_inputs_mask"].unsqueeze(0).to(device),
                        m_low=env_sample["low_inputs_mask"].unsqueeze(0).to(device),
                    )

        logits = logits.permute(0, 2, 3, 1)
        labels, unk_masks = segmentation_ground_truths(sample)

        if cfg["MODEL"]["num_classes"] == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > cfg["MODEL"]["threshold"]).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)

        loss = model.loss_fn["mean"](
            logits.reshape(-1, cfg["MODEL"]["num_classes"]),
            labels.to(device).reshape(-1).long()
        )

        if unk_masks is not None:
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            probs = probs.view(-1, cfg["MODEL"]["num_classes"])[unk_masks.view(-1)].cpu().numpy()
            labels_flat = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            preds = predicted.view(-1).cpu().numpy()
            probs = probs.view(-1, cfg["MODEL"]["num_classes"]).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()

        loss = loss.view(-1).cpu().detach().numpy()

        tot_preds.append(preds); tot_labels.append(labels_flat)
        tot_losses.append(loss); tot_probs.append(probs)

        region = [img_name_info["region"]] * len(preds)
        fwi = [img_name_info["fwinx_mean"]] * len(preds)
        tot_regions.append(region); tot_fwi.append(fwi)

    # Concatenate + save arrays
    predicted_classes = np.concatenate(tot_preds)
    target_classes = np.concatenate(tot_labels)
    losses = np.concatenate(tot_losses)
    probs_classes = np.concatenate(tot_probs)
    np.save(output_dir / f"{cfg['split']}_probs.npy", probs_classes)
    np.save(output_dir / f"{cfg['split']}_target.npy", target_classes)
    np.save(output_dir / f"{cfg['split']}_preds.npy", predicted_classes)
    np.save(output_dir / f"{cfg['split']}_losses_debug.npy", predicted_classes)

    regions = np.concatenate(tot_regions)
    fwis = np.concatenate(tot_fwi)
    np.save(output_dir / f"{cfg['split']}_regions.npy", regions)
    np.save(output_dir / f"{cfg['split']}_fwi.npy", fwis)

    # Metrics (unchanged)
    eval_metrics = get_classification_metrics(
        predicted=predicted_classes,
        labels=target_classes,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        unk_masks=None,
    )
    micro_auc, macro_auc, class_auc = get_pr_auc_scores(
        scores=probs_classes,
        labels=target_classes,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        output_dir=output_dir,
    )

    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics["micro"]
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics["macro"]
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]

    metrics = {
        "loss": losses.mean(),
        "macro_Accuracy": macro_acc,
        "macro_Precision": macro_precision,
        "macro_Recall": macro_recall,
        "macro_F1": macro_F1,
        "macro_IOU": macro_IOU,
        "macro_AUC": macro_auc,
        "micro_Accuracy": micro_acc,
        "micro_Precision": micro_precision,
        "micro_Recall": micro_recall,
        "micro_F1": micro_F1,
        "micro_IOU": micro_IOU,
        "micro_AUC": micro_auc,
        "fire_Accuracy": class_acc[1],
        "fire_Precision": class_precision[1],
        "fire_Recall": class_recall[1],
        "fire_F1": class_F1[1],
        "fire_IOU": class_IOU[1],
        "fire_AUC": class_auc[1],
    }

    global_metrics_path = output_dir / f"{cfg['split']}_metrics.txt"
    with open(global_metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    log.info(f"[eval] Wrote global metrics to: {global_metrics_path.resolve()}")

    # (Per-region block omitted for brevity—keep your current version if needed.)

    # Manifest
    manifest = output_dir / "MANIFEST.txt"
    with open(manifest, "w") as mf:
        for p in [
            output_dir / f"{cfg['split']}_metrics.txt",
            output_dir / f"{cfg['split']}_probs.npy",
            output_dir / f"{cfg['split']}_target.npy",
            output_dir / f"{cfg['split']}_preds.npy",
            output_dir / f"{cfg['split']}_losses_debug.npy",
            output_dir / f"{cfg['split']}_regions.npy",
            output_dir / f"{cfg['split']}_fwi.npy",
        ]:
            try:
                size = p.stat().st_size
                mf.write(f"{p.resolve()}  {size} bytes\n")
            except FileNotFoundError:
                mf.write(f"{p.resolve()}  [MISSING]\n")
    log.info(f"[eval] Manifest written: {manifest.resolve()}")

    print("Evaluation completed!")
    print(metrics)


if __name__ == "__main__":
    evaluate()
