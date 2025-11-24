"""Script for launching PL training."""
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from DeepSatModels.utils.config_files_utils import copy_yaml
from DeepSatModels.utils.torch_utils import get_device
from src.constants import CONFIG_PATH
from src.data import get_data
from src.data.Canada.callback import FWICallback, WeightLossCallback, SwitchAllCallback
from src.models import get_model
from src.utils.torch_utils import load_from_checkpoint
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.getcwd())

def _corr2(a,b):
    # a,b: [N,H,W]
    am = a.mean(dim=(1,2), keepdim=True); bm = b.mean(dim=(1,2), keepdim=True)
    an = a - am; bn = b - bm
    num = (an*bn).mean(dim=(1,2))
    den = (an.pow(2).mean(dim=(1,2)).sqrt() * bn.pow(2).mean(dim=(1,2)).sqrt() + 1e-6)
    return (num/den).mean().item()

# ---- Sanity probe (run once before training) -------------------------------
import torch
import torch.nn.functional as F

MSCLIP_MEANS = torch.tensor([925.161,1183.128,1338.041,1667.254,2233.633,2460.96,2555.569,2619.542,2406.497,1841.645], dtype=torch.float32)
MSCLIP_STDS  = torch.tensor([1205.586,1223.713,1399.638,1403.298,1378.513,1434.924,1491.141,1454.089,1473.248,1365.08], dtype=torch.float32)

def _corr2(a,b):
    am = a.mean(dim=(1,2), keepdim=True); bm = b.mean(dim=(1,2), keepdim=True)
    an = a - am; bn = b - bm
    num = (an*bn).mean(dim=(1,2))
    den = (an.pow(2).mean(dim=(1,2)).sqrt() * bn.pow(2).mean(dim=(1,2)).sqrt() + 1e-6)
    return (num/den).mean().item()

@torch.no_grad()
def sanity_probe_once(datamodule, model, device, use_msclip_stats=True):
    print("\n========== [SANITY PROBE] ==========")
    batch = next(iter(datamodule.train_dataloader()))
    x = batch["inputs"]         # [B, T, C, H, W]  (already normalized)
    y = batch["labels"]         # e.g. [B, 1, 25, 25]
    doy = batch.get("doy", None)

    # Move to device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).float()
    if doy is not None:
        doy = doy.to(device, non_blocking=True)

    print(f"[inputs] shape={tuple(x.shape)} dtype={x.dtype} dev={x.device} finite={torch.isfinite(x).all().item()}")
    print(f"[labels] shape={tuple(y.shape)} dtype={y.dtype} unique={torch.unique(y).detach().cpu().numpy()[:10]} finite={torch.isfinite(y).all().item()}")

    # Basic stats on normalized inputs (per-channel)
    B,T,C,H,W = x.shape
    xn = x.view(-1, C, H, W)
    ch_mean = xn.mean(dim=(0,2,3))
    ch_std  = xn.std(dim=(0,2,3))
    print("[inputs] per-channel mean (norm):", [round(v.item(),3) for v in ch_mean])
    print("[inputs] per-channel std  (norm):", [round(v.item(),3) for v in ch_std])

    # Heuristic warnings if far from ~0/1
    off = [(abs(m.item()), s.item()) for m,s in zip(ch_mean, ch_std)]
    if any(m > 0.6 or s < 0.5 or s > 1.6 for m,s in off):
        print("[warn] Channel norms look far from ~N(0,1). Domain shift or wrong stats could hurt convergence.")

    # Optional: de-normalize to reflectance-like for band-order heuristics
    if use_msclip_stats and C == 10:
        means = MSCLIP_MEANS.to(x.device).view(1,1,C,1,1)
        stds  = MSCLIP_STDS.to(x.device).view(1,1,C,1,1)
        x_ref = x * stds + means
        xr = x_ref.view(-1, C, H, W)
        B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 = [xr[:,i] for i in range(10)]
        B4c = B4.clamp_min(0); B8c = B8.clamp_min(0)
        ndvi = (B8c - B4c) / (B8c + B4c + 1e-6)
        print(f"[band-order] NDVI mean/std/min/max: {ndvi.mean().item():.3f} {ndvi.std().item():.3f} {ndvi.min().item():.3f} {ndvi.max().item():.3f}")
        print("[band-order] corr(B8,B8A):", f"{_corr2(B8,B8A):.3f}", " corr(B11,B12):", f"{_corr2(B11,B12):.3f}")
        if _corr2(B8,B8A) < 0.8 or _corr2(B11,B12) < 0.8:
            print("[warn] Band correlations lower than usual; double-check band order mapping.")

    # ---- Robust forward call (LightningModule or plain nn.Module) ----
    logits = None
    # Try: model(batch_dict) style
    try:
        logits = model(x, doy)                      # positional args (matches your FactorizeModel)
    except TypeError:
        try:
            logits = model(x)                       # LightningModule.forward(batch_inputs_only)
        except Exception:
            # If it's a LM wrapper with .model inside:
            if hasattr(model, "model"):
                try:
                    logits = model.model(x, doy)    # underlying nn.Module with (x, doy)
                except TypeError:
                    logits = model.model(x)
            # Last resort: try named attribute self.model.image_encoder etc… but usually above succeeds.

    if logits is None:
        raise RuntimeError("Could not invoke model forward with any supported signature.")

    print(f"[forward] logits shape={tuple(logits.shape)} dtype={logits.dtype} finite={torch.isfinite(logits).all().item()}")

    # Ensure we compare at label size
    Lh, Lw = y.shape[-2], y.shape[-1]
    if logits.shape[-2:] != (Lh, Lw):
        print(f"[note] Resizing logits {logits.shape[-2:]} -> label size {(Lh,Lw)} for metrics check.")
        logits = F.interpolate(logits, size=(Lh, Lw), mode="bilinear", align_corners=False)

    # Quick binary IoU (adjust for multi-class if needed)
    if logits.shape[1] == 1 or logits.shape[1] == 2:
        if logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits[:, 0])
        y_bin = (y > 0.5).float()
        # Simple 0.5 threshold preview metric
        preds = (probs > 0.5).float()
        inter = (preds * y_bin).sum()
        union = preds.sum() + y_bin.sum() - inter + 1e-6
        iou = (inter / union).item()
        pos_ratio = y_bin.mean().item()
        print(f"[labels] pos_ratio={pos_ratio:.6f}")
        print(f"[one-batch IOU @0.5]={iou:.4f}")
    else:
        print("[info] multi-class: add per-class IoU here if needed.")

    print("=====================================\n")


@hydra.main(version_base=None, config_path=str(CONFIG_PATH))
def train_and_evaluate(cfg: DictConfig):
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
    #device = get_device(local_device_ids, allow_cpu=False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))   # torchrun sets this
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    seed_everything(cfg["SET-UP"]["seed"])

    datamodule = get_data(cfg)
    cfg["SOLVER"]["num_steps_train"] = len(datamodule.train_dataloader())

    print("RANK", os.getenv("RANK"), "LOCAL_RANK", os.getenv("LOCAL_RANK"),
      "WORLD_SIZE", os.getenv("WORLD_SIZE"),
      "current_device", torch.cuda.current_device())

    # Initialize model & Load weights
    net = get_model(cfg, device)
    if checkpoint:
        load_from_checkpoint(net, checkpoint)

    # Sanity probe (run once)
    #try:
    #    sanity_probe_once(datamodule, net.to(device), device, use_msclip_stats=True)
    #except Exception as e:
    #    print("[SANITY PROBE FAILED]", repr(e))
    #    raise
    
    # Set-up model checkpoint & callbacks
    checkpoint_callback_IoU = ModelCheckpoint(
        monitor="fire_F1",  # Not sure name macro/IOU
        dirpath=save_path,
        filename=f"{arch_name}-" + "{epoch:02d}-f1-{fire_F1:.2f}",
        mode="max",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    checkpoint_callback_step = ModelCheckpoint(
        dirpath=save_path,
        filename=f"{arch_name}-" + "{epoch:02d}-step-{step:.2f}",
        save_top_k=-1,
        every_n_train_steps=save_steps,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback_IoU, checkpoint_callback_step]
    reload_dataloaders_every_n_epochs = 0
    if "pos_epochs" in cfg["DATASETS"]["train"]:
        switch_callback = SwitchAllCallback(config=cfg)
        callbacks.append(switch_callback)
        reload_dataloaders_every_n_epochs = 1

    if "fwi_ths" in cfg["DATASETS"]["train"]:
        fwi_callback = FWICallback(config=cfg)
        callbacks.append(fwi_callback)
        reload_dataloaders_every_n_epochs = 1

    if "weights" in cfg["SOLVER"]:
        loss_callback = WeightLossCallback(config=cfg)
        callbacks.append(loss_callback)

    # Set-up Wandb logger
    wandb_logger = WandbLogger(
        project=cfg["CHECKPOINT"]["wandb_project"],
        entity=cfg["CHECKPOINT"]["wandb_user"],
        name=cfg["CHECKPOINT"]["experiment_name"],
        group =cfg["CHECKPOINT"]["group"],
    )

    # Copy the config file to the save_path and wandb
    copy_yaml(cfg)
    wandb_logger.log_hyperparams(cfg)

    print("test1")
    if False:   
        running_counts = Counter()
        for batch in tqdm(datamodule.train_dataloader()):
            doy = batch["doy"]
            doy = doy[:,:,0,0,0].int().view(-1)

            vals, counts = torch.unique(doy, return_counts=True)

            running_counts.update(dict(zip(vals.tolist(), counts.tolist())))

        for batch in tqdm(datamodule.val_dataloader()):
            doy = batch["doy"]
            doy = doy[:,:,0,0,0].int().view(-1)

            vals, counts = torch.unique(doy, return_counts=True)

            running_counts.update(dict(zip(vals.tolist(), counts.tolist())))
        
        for batch in tqdm(datamodule.test_dataloader()):
            doy = batch[0]["doy"]
            doy = doy[:,:,0,0,0].int().view(-1)

            vals, counts = torch.unique(doy, return_counts=True)

            running_counts.update(dict(zip(vals.tolist(), counts.tolist())))
        
        print(running_counts)
        sys.exit(0)

    print("test2")

    # Set-up Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,                 
        num_nodes=1,
        #strategy="ddp_find_unused_parameters_true",            # <-- "ddp" , "ddp_find_unused_parameters_true"
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        accumulate_grad_batches=cfg["SOLVER"]["accumulate_grad_batches"],
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
    )

    # Start training
    trainer.fit(model=net, datamodule=datamodule)

if __name__ == "__main__":
    train_and_evaluate()