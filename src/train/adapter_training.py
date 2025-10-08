import os
from pathlib import Path
import pickle
import glob
import json
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from DeepSatModels.utils.config_files_utils import copy_yaml
from DeepSatModels.utils.torch_utils import get_device
from src.constants import CONFIG_PATH

import sys
sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model


class L1C2L2AAdapter(nn.Module):
    """Linear transform in CLIP embedding space (DxD)."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x


def load_stats(mean_path, std_path):
    with open(mean_path, "r") as f:
        mean_json = json.load(f)
    with open(std_path, "r") as f:
        std_json = json.load(f)
    mean = torch.tensor(list(mean_json.values()), dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(list(std_json.values()), dtype=torch.float32).view(-1, 1, 1)
    return mean, std


def normalize_tensor(x, mean, std):
    mean, std = mean.to(x.device), std.to(x.device)
    if mean.shape[0] != x.shape[0]:
        mean, std = mean[:x.shape[0]], std[:x.shape[0]]
    return (x - mean) / std


def cosine_loss(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1 - (a * b).sum(dim=-1).mean()


class WorldStratPairsStream(IterableDataset):
    def __init__(self, parquet_dir, mean, std, max_rows=None):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        assert self.files, f"No parquet files in {parquet_dir}"
        self.mean, self.std = mean, std
        self.max_rows = max_rows

    def __iter__(self):
        count = 0
        for f in self.files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=1):
                row = batch.to_pandas().iloc[0]
                yield self._row_to_sample(row)
                count += 1
                if self.max_rows and count >= self.max_rows:
                    return

    def _row_to_sample(self, row):
        ten_L1C = torch.from_numpy(pickle.loads(row["10x_L1C"])).float()
        twenty_L1C = torch.from_numpy(pickle.loads(row["20x_L1C"])).float()
        ten_L2A = torch.from_numpy(pickle.loads(row["10x_L2A"])).float()
        twenty_L2A = torch.from_numpy(pickle.loads(row["20x_L2A"])).float()

        twenty_L1C = F.interpolate(twenty_L1C.unsqueeze(0), size=ten_L1C.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        twenty_L2A = F.interpolate(twenty_L2A.unsqueeze(0), size=ten_L2A.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)

        x_L1C = torch.cat([ten_L1C, twenty_L1C], 0)
        x_L2A = torch.cat([ten_L2A, twenty_L2A], 0)

        x_L1C = F.interpolate(x_L1C.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        x_L2A = F.interpolate(x_L2A.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        x_L1C = normalize_tensor(x_L1C, self.mean, self.std)
        x_L2A = normalize_tensor(x_L2A, self.mean, self.std)
        return x_L1C, x_L2A



def train_adapter(cfg, msclip_encoder, adapter, train_loader, val_loader, device, wandb_logger):
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg["SOLVER"]["lr"], weight_decay=cfg["SOLVER"]["weight_decay"])
    num_epochs = cfg["SOLVER"]["num_epochs"]

    msclip_encoder.eval().to(device)
    for p in msclip_encoder.parameters():
        p.requires_grad = False
    adapter.to(device)

    for epoch in range(num_epochs):
        adapter.train()
        train_loss = []
        for xb_L1C, xb_L2A in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train"):
            xb_L1C, xb_L2A = xb_L1C.to(device), xb_L2A.to(device)
            with torch.no_grad():
                E_L1C = msclip_encoder.get_patch_embeddings(xb_L1C)[:, 1:, :]
                E_L2A = msclip_encoder.get_patch_embeddings(xb_L2A)[:, 1:, :]
            e1 = E_L1C.reshape(-1, E_L1C.shape[-1])
            e2 = E_L2A.reshape(-1, E_L2A.shape[-1])

            e1c = adapter(e1)
            #loss_cos = cosine_loss(e1c, e2)
            loss_mse = F.mse_loss(e1c, e2)
            loss = loss_mse  

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())

        mean_train_loss = np.mean(train_loss)

        # Validation
        adapter.eval()
        val_loss, val_acc = [], []
        with torch.no_grad():
            for xb_L1C, xb_L2A in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Val"):
                xb_L1C, xb_L2A = xb_L1C.to(device), xb_L2A.to(device)
                E_L1C = msclip_encoder.get_patch_embeddings(xb_L1C)[:, 1:, :]
                E_L2A = msclip_encoder.get_patch_embeddings(xb_L2A)[:, 1:, :]
                e1 = E_L1C.reshape(-1, E_L1C.shape[-1])
                e2 = E_L2A.reshape(-1, E_L2A.shape[-1])
                e1c = adapter(e1)
                l = cosine_loss(e1c, e2) + 0.1 * F.mse_loss(e1c, e2)
                val_loss.append(l.item())

        mean_val_loss = np.mean(val_loss)

        print(f"Epoch {epoch+1}: train_loss={mean_train_loss:.4f}, val_loss={mean_val_loss:.4f}")
        wandb_logger.log_metrics({"train/loss": mean_train_loss, "val/loss": mean_val_loss}, step=epoch)

    return adapter


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="adapter_config")
def train_adapter_hydra(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    seed_everything(cfg["SETUP"]["seed"])
    device = get_device(cfg["SETUP"]["local_device_ids"], allow_cpu=False)

    wandb_logger = WandbLogger(
        project=cfg["CHECKPOINT"]["wandb_project"],
        entity=cfg["CHECKPOINT"]["wandb_user"],
        name=cfg["CHECKPOINT"]["experiment_name"],
    )
    copy_yaml(cfg)
    wandb_logger.log_hyperparams(cfg)

    print("Loading MS-CLIP...")
    msclip_model, _, _ = build_model(
        model_name=cfg["MODEL"]["msclip_model_name"],
        pretrained=True,
        ckpt_path=cfg["MODEL"]["msclip_ckpt"],
        device=device,
        channels=cfg["MODEL"]["channels"],
    )
    encoder = msclip_model.image_encoder
    embed_dim = encoder.get_patch_embeddings(torch.zeros(1, cfg["MODEL"]["channels"], 224, 224)).shape[-1]

    adapter = L1C2L2AAdapter(dim=embed_dim, dropout=cfg["SOLVER"]["dropout"])

    mean, std = load_stats(cfg["DATASETS"]["kwargs"]["mean_file"], cfg["DATASETS"]["kwargs"]["std_file"])

    train_ds = WorldStratPairsStream(cfg["DATASETS"]["train"]["data_dir"], mean, std)
    val_ds = WorldStratPairsStream(cfg["DATASETS"]["eval"]["data_dir"], mean, std, max_rows=cfg["DATASETS"].get("val_max_rows", 1000))
    collate = lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]))
    train_loader = DataLoader(train_ds, batch_size=cfg["DATASETS"]["train"]["batch_size"], collate_fn=collate, num_workers=cfg["DATASETS"]["train"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["DATASETS"]["eval"]["batch_size"], collate_fn=collate, num_workers=cfg["DATASETS"]["eval"]["num_workers"])

    adapter = train_adapter(cfg, encoder, adapter, train_loader, val_loader, device, wandb_logger)

    Path(cfg["CHECKPOINT"]["save_path"]).mkdir(parents=True, exist_ok=True)
    out_ckpt = os.path.join(cfg["CHECKPOINT"]["save_path"], "l1c2l2a_linear.pt")
    torch.save(adapter.state_dict(), out_ckpt)
    print(f" Saved adapter to {out_ckpt}")


if __name__ == "__main__":
    train_adapter_hydra()
