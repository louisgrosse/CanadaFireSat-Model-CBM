import math
from pathlib import Path
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import sys

# --- Lightning & logging ---
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils.ssl4eos12_dataset import (
    SSL4EOS12Dataset, collate_fn,
    S2L1C_MEAN, S2L1C_STD
)

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MSClipLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        initial_temperature: float = 0.07,
        patch_alignment: bool = True,
        pacl_weight: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build MS-CLIP exactly as released by IBM (multispectral image tower + tokenizer)
        self.model, _preprocess, self.tokenizer = build_model(model_name="Llama3-MS-CLIP-Base")

        # Contrastive loss pieces
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / initial_temperature, dtype=torch.float32))
        self.lr = lr
        self.weight_decay = weight_decay
        self.patch_alignment = patch_alignment
        self.pacl_weight = pacl_weight

    @staticmethod
    def _clip_loss(img_emb, txt_emb, logit_scale):
        # Normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        # Similarities
        logits_per_image = logit_scale.exp() * img_emb @ txt_emb.t()
        logits_per_text  = logits_per_image.t()
        targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        loss_i = nn.functional.cross_entropy(logits_per_image, targets)
        loss_t = nn.functional.cross_entropy(logits_per_text,  targets)
        return 0.5 * (loss_i + loss_t), logits_per_image

    def forward(self, images, captions):
        tokens = self.tokenizer(captions).to(self.device)
        out = self.model.encode_image(images)       # could be Tensor or tuple
        txt_emb = self.model.encode_text(tokens)

        cls_emb, patch_emb = None, None
        if isinstance(out, tuple):
            # Per their metrics code: index 0 is the global image embedding
            cls_emb = out[0]
            # If a second element exists and looks like patch tokens, use it for patch alignment
            if len(out) > 1 and hasattr(out[1], "dim") and out[1].dim() == 3:
                patch_emb = out[1]                  # [B, N, D]
        else:
            cls_emb = out

        # For the patch loss we’ll mean-pool tokens to [B, D]
        if patch_emb is not None and patch_emb.dim() == 3:
            patch_emb = patch_emb.mean(dim=1)

        return cls_emb, txt_emb, patch_emb

    def training_step(self, batch, batch_idx):
        images = batch["S2L1C"]
        captions = batch["captions"]
        if hasattr(captions, "ndim") and captions.ndim == 2 and captions.shape[1] >= 1:
            captions = [row[-1] for row in captions]
        elif isinstance(captions, list):
            captions = [c[-1] if isinstance(c, (list, tuple)) else c for c in captions]
        captions = ["a satellite image" if (c is None or (isinstance(c, float) and math.isnan(c))) else str(c) for c in captions]
        images = images.to(self.device, non_blocking=True)

        img_emb, txt_emb, patch_emb = self(images, captions)
        loss_cls, logits = self._clip_loss(img_emb, txt_emb, self.logit_scale)

        if self.patch_alignment and patch_emb is not None:
            loss_patch, _ = self._clip_loss(patch_emb, txt_emb, self.logit_scale)
            loss = self.pacl_weight * loss_patch + (1.0 - self.pacl_weight) * loss_cls
        else:
            loss = loss_cls

        # metrics...
        # (unchanged)
        return loss

    def validation_step(self, batch, batch_idx):
        # same caption handling as above...
        img_emb, txt_emb, patch_emb = self(images, captions)
        loss_cls, logits = self._clip_loss(img_emb, txt_emb, self.logit_scale)
        if self.patch_alignment and patch_emb is not None:
            loss_patch, _ = self._clip_loss(patch_emb, txt_emb, self.logit_scale)
            loss = self.pacl_weight * loss_patch + (1.0 - self.pacl_weight) * loss_cls
        else:
            loss = loss_cls
        # metrics...
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["S2L1C"]
        captions = batch["captions"]
        if hasattr(captions, "ndim") and captions.ndim == 2 and captions.shape[1] >= 1:
            captions = [row[-1] for row in captions]
        elif isinstance(captions, list):
            captions = [c[-1] if isinstance(c, (list, tuple)) else c for c in captions]
        captions = ["a satellite image" if (c is None or (isinstance(c, float) and math.isnan(c))) else str(c) for c in captions]
        images = images.to(self.device, non_blocking=True)

        img_emb, txt_emb, patch_emb = self(images, captions)
        loss_cls, logits = self._clip_loss(img_emb, txt_emb, self.logit_scale)
        if self.patch_alignment and patch_emb is not None:
            loss_patch, _ = self._clip_loss(patch_emb, txt_emb, self.logit_scale)
            loss = self.pacl_weight * loss_patch + (1.0 - self.pacl_weight) * loss_cls
        else:
            loss = loss_cls

        targets = torch.arange(logits.size(0), device=logits.device)
        acc_i = (logits.argmax(dim=1) == targets).float().mean()
        acc_t = (logits.t().argmax(dim=1) == targets).float().mean()
        acc = 0.5 * (acc_i + acc_t)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val/acc",  acc,  on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt

def make_dataloaders(cfg):
    data_root = Path(cfg["DATA"]["root"])
    train_tf = transforms.Compose([
        transforms.RandomCrop(cfg["MODEL"]["image_size"]),
        transforms.Normalize(S2L1C_MEAN, S2L1C_STD),
    ])
    val_tf = transforms.Compose([
        transforms.CenterCrop(cfg["MODEL"]["image_size"]),
        transforms.Normalize(S2L1C_MEAN, S2L1C_STD),
    ])
    train_set = SSL4EOS12Dataset(
        data_dir=str(data_root / "train"),
        split_file=str(data_root / cfg["DATA"]["train_split"]),
        modalities=["S2L1C", "captions"],
        transform=train_tf,
        concat=False,
        single_timestamp=True,
    )
    val_set = SSL4EOS12Dataset(
        data_dir=str(data_root / "val"),
        split_file=str(data_root / cfg["DATA"]["val_split"]),
        modalities=["S2L1C", "captions"],
        transform=val_tf,
        concat=False,
        single_timestamp=True,
    )
    train_loader = DataLoader(train_set, batch_size=cfg["TRAIN"]["batch_size_zarr"],
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["TRAIN"]["batch_size_zarr"],
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True)
    return train_loader, val_loader

def main():
    with open("/home/grosse/CanadaFireSat-Model-CBM/configs/fineTuneClip.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # --- W&B logger (Lightning) ---
    wandb_logger  = WandbLogger(
        project=cfg["CHECKPOINT"]["project"],
        entity=cfg["CHECKPOINT"]["user"],
        name=cfg["CHECKPOINT"]["experiment_name"],
        group=cfg["CHECKPOINT"]["group"],
    )

    train_loader, val_loader = make_dataloaders(cfg)

    module = MSClipLightningModule(
        lr=cfg["TRAIN"]["lr"],
        weight_decay=cfg["TRAIN"]["weight_decay"],
        initial_temperature=cfg["MODEL"]["initial_temperature"],
        patch_alignment=bool(cfg["MODEL"]["patch_alignment"]),
        pacl_weight=cfg["TRAIN"]["pacl_weight"],
    )

    save_dir = Path(cfg["CHECKPOINT"]["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="msclip_l1c-{epoch:02d}-{global_step:07d}",
        save_top_k=2,
        monitor="val/loss",
        mode="min",
        save_on_train_epoch_end=False,
        every_n_train_steps=cfg["CHECKPOINT"]["save_steps"],
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg["TRAIN"]["max_epochs"],
        logger=wandb_logger,
        log_every_n_steps=cfg["CHECKPOINT"]["train_metrics_steps"],
        callbacks=[ckpt_cb, lr_cb],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )

    trainer.fit(module, train_loader, val_loader)

    # Save a lightweight PyTorch ckpt alongside Lightning’s
    torch.save(
        {
            "model_state": module.model.state_dict(),
            "logit_scale": module.logit_scale.detach().cpu(),
            "cfg": cfg,
        },
        str(save_dir / "msclip_l1c_finetune.pt"),
    )

if __name__ == "__main__":
    main()