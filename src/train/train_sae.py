#!/usr/bin/env python3
import os
import math
import random
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from einops import rearrange
import numpy as np

from src.constants import CONFIG_PATH
from overcomplete.sae import TopKSAE

# ------------------------------
# Utilities
# ------------------------------

def _flatten_tokens(arr: np.ndarray) -> np.ndarray:
    # arr: (N, D, H, W) -> (N*H*W, D)
    N, D, H, W = arr.shape
    arr = rearrange(arr, 'n d h w -> (n h w) d')
    return arr

@torch.no_grad()
def estimate_feature_stats(memmap_path: str, max_tokens: int = 200_000, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample up to max_tokens patches to estimate per-dim mean/std."""
    rng = np.random.default_rng(seed)
    mm = np.load(memmap_path, mmap_mode="r")  # (N, D, Hp, Wp)
    N, D, Hp, Wp = mm.shape
    tokens_per_item = Hp * Wp
    total_tokens = N * tokens_per_item
    sample_idx = rng.choice(total_tokens, size=min(max_tokens, total_tokens), replace=False)

    # Welford
    mean = torch.zeros(D)
    M2 = torch.zeros(D)
    n = 0
    for flat in sample_idx:
        n_item = flat // tokens_per_item
        r = flat % tokens_per_item
        y, x = divmod(r, Wp)
        v = torch.tensor(mm[n_item, :, y, x], dtype=torch.float32)
        n += 1
        delta = v - mean
        mean += delta / n
        delta2 = v - mean
        M2 += delta * delta2
    var = M2 / max(n - 1, 1)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean, std

class SAEFeatures(Dataset):
    def __init__(self, npy_path: str, mean: Optional[torch.Tensor], std: Optional[torch.Tensor], flatten_tokens: bool = True):
        self.path = Path(npy_path)
        self.mm = np.load(self.path, mmap_mode="r")  # (N, D, Hp, Wp)
        self.N, self.D, self.Hp, self.Wp = self.mm.shape
        self.flatten = flatten_tokens
        self.tokens_per_item = self.Hp * self.Wp
        self.total = self.N * self.tokens_per_item if self.flatten else self.N
        self.mean = mean
        self.std = std

    def __len__(self): return self.total

    def __getitem__(self, i):
        if self.flatten:
            n = i // self.tokens_per_item
            r = i % self.tokens_per_item
            y, x = divmod(r, self.Wp)
            v = self.mm[n, :, y, x]  # (D,)
        else:
            v = self.mm[i]           # (D, Hp, Wp)
        t = torch.from_numpy(np.array(v, copy=True)).float()
        if self.mean is not None and self.std is not None:
            t = (t - self.mean) / (self.std + 1e-6)
        return t

def get_decoder_weight(model: nn.Module) -> torch.Tensor:
    # Try common accessors; fall back to attribute search.
    if hasattr(model, "get_dictionary"):
        return model.get_dictionary()
    if hasattr(model, "dictionary"):
        dic = getattr(model, "dictionary")
        if hasattr(dic, "_weights"):
            return dic._weights
        if hasattr(dic, "weight"):
            return dic.weight
    # last resort: search parameters
    for n, param in model.named_parameters():
        if any(k in n.lower() for k in ["dictionary", "decoder", "dec", "dict"]):
            return param
    raise AttributeError("Could not locate decoder/dictionary weights on the SAE model.")

def initialize_encoder_from_decoder(sae_model: TopKSAE):
    # weight-tie init (common trick used in the repo)
    W_dec = sae_model.dictionary._weights
    enc_linear = sae_model.encoder.final_block[0]
    import torch
    with torch.no_grad():
        enc_linear.weight.copy_(W_dec)
        enc_linear.bias.zero_()

# AuxK loss (as used in the repo you showed)
def top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary, penalty=1.0, scale=None, shift=None):
    """
    Predict residual with top 50% of the non-chosen codes (using pre-activations minus chosen codes).
    Loss = ||x - x_hat||^2 + penalty * || (x - x_hat) - residual_hat ||^2
    """
    residual = x - x_hat
    mse = residual.square().mean()

    pre_codes = torch.relu(pre_codes)
    pre_codes = pre_codes - codes  # non-chosen portion

    auxiliary_topk = torch.topk(pre_codes, k=max(1, pre_codes.shape[1] // 2), dim=1)
    pre_codes = torch.zeros_like(codes).scatter(-1, auxiliary_topk.indices, auxiliary_topk.values)

    residual_hat = pre_codes @ dictionary  # [B, K] @ [K, D] -> [B, D]
    if scale is not None and shift is not None:
        residual_hat = (residual_hat - shift) / (scale + 1e-8)
    aux_mse = (residual - residual_hat).square().mean()

    return mse, aux_mse

# ------------------------------
# LightningModule wrapping the SAE
# ------------------------------
class SAELightning(pl.LightningModule):
    def __init__(self, cfg: Dict, feat_mean: torch.Tensor, feat_std: torch.Tensor):
        super().__init__()
        self.save_hyperparameters(OmegaConf.create(cfg))  # for checkpointing

        sae_cfg = cfg["SAE"]
        self.model = TopKSAE(
            input_shape=sae_cfg["d_in"],
            nb_concepts=sae_cfg["nb_concepts"],
            top_k=sae_cfg["top_k"],
            device=sae_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        initialize_encoder_from_decoder(self.model)

        tr = cfg["TRAIN"]
        self.lr = tr["lr"]
        self.weight_decay = tr.get("weight_decay", 0.0)
        self.penalty_auxk = tr.get("lambda_auxk", 1.0)
        self.lambda_l1 = tr.get("lambda_l1_codes", 1e-3)
        self.lambda_usage = tr.get("lambda_usage", 1e-3)

        self.resample_every_n_epochs = tr.get("resample_every_n_epochs", 1)
        self.resample_num_samples = tr.get("resample_num_samples", 100000)
        self.resample_batch_size = tr.get("resample_batch_size", 1)

        # Stats for standardization saved for inference
        self.register_buffer("feat_mean", feat_mean.clone())
        self.register_buffer("feat_std", feat_std.clone())

        # Track usage to estimate "dead"
        M = sae_cfg["nb_concepts"]
        self.register_buffer("usage_counts", torch.zeros(M))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)  # already standardized by dataset
        pre_codes, codes, x_hat = self.model(x)
        dictionary = get_decoder_weight(self.model)

        mse, aux_mse = top_k_auxiliary_loss(
            x, x_hat, pre_codes, codes, dictionary, penalty=self.penalty_auxk,
        )
        auxk_total = mse + self.penalty_auxk * aux_mse
        l1_codes = codes.abs().mean()

        # Usage balancing
        usage = (codes > 0).float().mean(dim=0) + 1e-8  # [K]
        p_hat = usage / usage.sum()
        uniform_log = math.log(1.0 / p_hat.numel())
        kl = torch.sum(p_hat * (torch.log(p_hat) - uniform_log))

        loss = auxk_total + self.lambda_l1 * l1_codes + self.lambda_usage * kl

        # Logging
        self.log("loss/total", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/mse", mse, on_step=True, on_epoch=True)
        self.log("loss/aux_mse", aux_mse, on_step=True, on_epoch=True)
        self.log("loss/l1_codes", l1_codes, on_step=True, on_epoch=True)
        self.log("loss/usage_kl", kl, on_step=True, on_epoch=True)

        # Running usage for dead estimate
        with torch.no_grad():
            self.usage_counts += (codes > 0).float().sum(dim=0).detach()

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Cosine with linear warmup ~3%
        steps_per_epoch = self.trainer.estimated_stepping_batches // max(1, self.trainer.max_epochs)
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup = max(1, int(0.03 * total_steps))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    @torch.no_grad()
    def _collect_usage_dead_mask(self, loader: DataLoader) -> torch.Tensor:
        device = self.device
        usage = None
        for xb in loader:
            xb = xb.to(device)
            pre_b, codes_b, _ = self.model(xb)
            fired = (codes_b > 0).to(torch.float32).sum(dim=0)
            if usage is None:
                usage = fired
            else:
                usage += fired
        dead_mask = (usage == 0)
        return dead_mask

    @torch.no_grad()
    def _resample_dead_codes(self, loader: DataLoader, dead_indices: torch.Tensor):
        """Mimics the trustworthy repo's resampling approach."""
        if dead_indices.numel() == 0:
            return 0

        device = self.device
        dataset = loader.dataset
        N = len(dataset)
        k = dead_indices.numel()

        # Sample candidates; score by recon MSE
        sampled_indices = random.sample(range(N), min(self.resample_num_samples, N))
        subset = Subset(dataset, sampled_indices)
        subset_loader = DataLoader(subset, batch_size=self.resample_batch_size, shuffle=False, num_workers=0)

        tot_indices = []
        tot_losses = []
        for i, batch in enumerate(subset_loader):
            x = batch.to(device)
            pre_codes, codes, x_hat = self.model(x)
            err = (x - x_hat).reshape(x.shape[0], -1).pow(2).mean(dim=1)
            start = i * self.resample_batch_size
            end = start + x.shape[0]
            tot_indices.extend(sampled_indices[start:end])
            tot_losses.append(err.detach().cpu())

        tot_losses = torch.cat(tot_losses, dim=0)
        probs = tot_losses / tot_losses.sum()
        chosen_local = torch.multinomial(probs, num_samples=k, replacement=False)
        chosen_dataset_indices = [tot_indices[i] for i in chosen_local]

        dec_weight_param = get_decoder_weight(self.model)
        # Encoder params (assumes Linear final block)
        enc_linear = self.model.encoder.final_block[0]
        enc_weight_param = enc_linear.weight
        enc_bias_param = enc_linear.bias

        # target scale = mean norm of alive enc rows
        alive_mask = torch.ones(dec_weight_param.shape[0], dtype=torch.bool, device=dec_weight_param.device)
        alive_mask[dead_indices] = False
        alive_norm = enc_weight_param[alive_mask, :].norm(dim=1)
        mean_alive = alive_norm.mean().clamp(min=1e-6)

        for dead_idx, ds_idx in zip(dead_indices.tolist(), chosen_dataset_indices):
            item = dataset[ds_idx]
            x = item.to(device) if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device, dtype=torch.float32)
            v = x.reshape(-1)
            if v.numel() != dec_weight_param.shape[1]:
                v = v[:dec_weight_param.shape[1]]
            v = v / (v.norm() + 1e-6) * mean_alive

            # Set decoder row
            dec_weight_param[dead_idx, :].copy_(v)
            # Set encoder row and bias
            enc_weight_param[dead_idx, :].copy_(v)
            enc_bias_param[dead_idx].zero_()

            # Zero optimizer moments (Adam)
            opt = self.trainer.optimizers[0]
            if isinstance(opt, torch.optim.Adam):
                for param, index in [(enc_weight_param, dead_idx), (enc_bias_param, dead_idx), (dec_weight_param, dead_idx)]:
                    state = opt.state.get(param, None)
                    if state and "exp_avg" in state and "exp_avg_sq" in state:
                        try:
                            state["exp_avg"][index].zero_()
                            state["exp_avg_sq"][index].zero_()
                        except Exception:
                            pass
        return k

    def on_train_epoch_end(self):
        # Dead-code resampling cadence
        if self.resample_every_n_epochs <= 0:
            return
        if (self.current_epoch + 1) % self.resample_every_n_epochs != 0:
            return
        loader = self.trainer.datamodule.train_dataloader()
        dead_mask = self._collect_usage_dead_mask(loader)
        dead_indices = torch.where(dead_mask)[0].to(self.device)
        dead_count_epoch = int(dead_mask.sum().item())
        dead_frac_epoch = dead_count_epoch / dead_mask.numel()
        self.log("monitor/dead_count_epoch", dead_count_epoch, prog_bar=True)
        self.log("monitor/dead_frac_epoch", dead_frac_epoch, prog_bar=True)
        revived = self._resample_dead_codes(loader, dead_indices)
        if revived > 0:
            self.log("monitor/resampled_dead", float(revived), prog_bar=True)

# ------------------------------
# DataModule (Lightning-style)
# ------------------------------
class ActDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.ds = None
        self.batch_size = cfg["TRAIN"]["batch_size"]
        self.num_workers = cfg["TRAIN"]["num_workers"]

    def setup(self, stage: Optional[str] = None):
        data = self.cfg["DATA"]
        self.ds = SAEFeatures(data["train_features"], self.mean, self.std, flatten_tokens=True)

        sae_cfg = self.cfg["SAE"]
        if sae_cfg.get("d_in") is not None:
            assert sae_cfg["d_in"] == self.ds.D, f"d_in={sae_cfg['d_in']} but dump has D={self.ds.D}"
        else:
            self.cfg["SAE"]["d_in"] = self.ds.D

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

# ------------------------------
# Hydra main
# ------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="sae_config.yaml")
def main(cfg: DictConfig):
    # Resolve nested interpolations
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # --- Stats for standardization (kept for CBM inference)
    feat_mean, feat_std = estimate_feature_stats(
        cfg["DATA"]["train_features"],
        max_tokens=cfg["TRAIN"].get("stats_max_tokens", 200_000),
        seed=cfg.get("SET-UP", {}).get("seed", 0),
    )

    # --- Lightning DataModule + Model
    dm = ActDataModule(cfg, feat_mean, feat_std)
    dm.setup()

    model = SAELightning(cfg, feat_mean, feat_std)

    # --- Logger & callbacks (trustworthy style)
    ckpt = cfg.get("CHECKPOINT", {})
    out = cfg["OUTPUT"]
    out_dir = Path(out["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project=ckpt.get("wandb_project", "sae"),
        entity=ckpt.get("wandb_user", None),
        name=ckpt.get("experiment_name", None),
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="sae-{epoch:03d}-{loss_total:.4f}",
        save_top_k=1,
        monitor="loss/total_epoch",
        mode="min",
        save_last=True,
    )

    # --- Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["TRAIN"]["nb_epochs"],
        logger=wandb_logger,
        callbacks=[lr_cb, ckpt_cb],
        log_every_n_steps=cfg["TRAIN"].get("log_every", 50),
        gradient_clip_val=1.0,
    )

    # --- Fit
    trainer.fit(model=model, datamodule=dm)

    # --- Save final checkpoint with mean/std for inference
    final_ckpt = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "D": cfg["SAE"]["d_in"],
        "nb_concepts": cfg["SAE"]["nb_concepts"],
        "top_k": cfg["SAE"]["top_k"],
        "feat_mean": feat_mean,    # tensors [D]
        "feat_std": feat_std,      # tensors [D]
    }
    torch.save(final_ckpt, out_dir / "topk_sae_final.pt")

    print(f"[done] Saved checkpoint to {out_dir / 'topk_sae_final.pt'}")

if __name__ == "__main__":
    main()
