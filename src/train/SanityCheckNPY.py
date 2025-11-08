# SanityCheckNPY.py — minimal changes to use NPY activation batches instead of SAE decoder

from typing import List, Optional, Dict, Any

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import hydra
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.train.train_sae import NpyActDataModule, NpyActivationDataset

from src.models.sae import plSAE
from src.utils.process_utils import save_activations_to_npy, save_labels_to_npy
import numpy as np
from src.constants import CONFIG_PATH
import sys

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Dict[Any, Any]:
    """Compute cosine similarity baseline between NPY activation maps and concept dictionaries."""

    if not HydraConfig.initialized():
        HydraConfig.instance().clear()
        HydraConfig().set_config(cfg)
    hydra_run_dir = HydraConfig.get().run.dir

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    # NPY activations datamodule (we'll use its train loader for batches of vectors)
    act_datamodule = NpyActDataModule(
        batch_size=cfg.sae_batch_size,
        train_npy_path=cfg.datamodule["train_path"],
        val_npy_path=cfg.datamodule["train_path"],
        test_npy_path=cfg.datamodule["train_path"]
    )

    align = cfg["ALIGN"]
    device = str(align.get("device", "cuda"))

    # Add "max" to CSV columns
    columns = ["dict", "mean", "std", "max", "N"]
    rows = []

    dataloader = act_datamodule.train_dataloader()

    for csv_name in align["csv_names"]:
        # 1) Load dictionary of concepts
        df = pd.read_csv(str(align["csv_phrases_path"]) + str(csv_name))
        phrases = df["concept"].astype(str).tolist()

        # 2) Build MS-CLIP & text embeddings
        model_, _, tokenizer = build_model(
            model_name=align["msclip_model_name"],
            pretrained=bool(align.get("pretrained", True)),
            ckpt_path=align.get("msclip_ckpt"),
            device=device,
            channels=10,
        )
        msclip = model_.eval().to(device)

        text_embs = _encode_phrases_msclip(
            phrases=phrases,
            model=msclip,
            tokenizer=tokenizer,
            batch_size=int(align.get("text_batch_size", 512)),
            device=device,
        )  # [N, Dt]

        vp = msclip.clip_base_model.model.visual  # maps visual -> text space

        # 3) Stream over activation batches and aggregate stats
        sum_top1 = 0.0
        sumsq_top1 = 0.0
        max_top1 = -float("inf")
        count_total = 0

        for batch in tqdm(dataloader):
            mean_b, std_b, max_b, n_b = _compute_alignment_stats(
                batch=batch,
                text_embs=text_embs,
                vision=vp,
                device=device,
            )
            # Aggregate: E[x] and E[x^2] across batches (population stats)
            sum_top1 += mean_b * n_b
            sumsq_top1 += (std_b ** 2 + mean_b ** 2) * n_b  # since E[x^2] = var + mean^2
            count_total += n_b
            if max_b > max_top1:
                max_top1 = max_b

        if count_total > 0:
            mean = sum_top1 / count_total
            var = max(0.0, (sumsq_top1 / count_total) - (mean ** 2))
            std = var ** 0.5
        else:
            mean, std = float("nan"), float("nan")
            max_top1 = float("nan")

        rows.append([csv_name, float(mean), float(std), float(max_top1), len(phrases)])
        print(f"[ALIGN] top1 cosine — mean={mean:.4f}, std={std:.4f}, max={max_top1:.4f}  {csv_name}")

    # 4) Save CSV in the Hydra run dir
    out_df = pd.DataFrame(rows, columns=columns)
    os.makedirs(hydra_run_dir, exist_ok=True)
    out_csv = os.path.join(hydra_run_dir, "/home/grosse/CanadaFireSat-Model-CBM/results/dictionnary/npy_vs_dicts_cosine.csv")
    out_df.to_csv(out_csv, index=False)
    log.info(f"Saved results to: {out_csv}")
    return {"csv_path": out_csv}


@torch.no_grad()
def _encode_phrases_msclip(phrases, model, tokenizer, batch_size, device):
    embs = []
    for i in range(0, len(phrases), batch_size):
        toks = tokenizer(phrases[i:i+batch_size]).to(device)
        z = model.inference_text(toks)          # [B, Dt]
        embs.append(F.normalize(z, dim=-1))
    return torch.cat(embs, dim=0)               # [N, Dt]


@torch.no_grad()
def _compute_alignment_stats(batch, text_embs, vision, device: str):
    """
    Compute per-batch cosine stats between activation vectors (from NPY dataloader)
    and the text embeddings for a given dictionary of concepts.
    Returns (mean, std, max, count) for top-1 cosine across concepts.
    """

    X = batch["inputs"]

    B, D, H ,W = X.shape

    X = X.permute(0, 2, 3, 1).contiguous().view(B * H * W, D).to(device)

    # Normalize and cosine sims vs text
    X = F.normalize(X, dim=1)                                  # [B, Dt]
    T = F.normalize(text_embs.to(device).float(), dim=1)       # [N, Dt]
    sims = X @ T.t()                                           # [B, N]

    top1 = sims.max(dim=1).values                              # [B]
    mean = top1.mean().item()
    std = top1.std(unbiased=False).item()
    max_val = top1.max().item()
    return mean, std, max_val, top1.numel()


@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="sae_config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
