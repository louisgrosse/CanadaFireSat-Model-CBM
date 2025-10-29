from typing import List, Optional, Dict, Any

import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from einops import rearrange
import os
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import logging
import numpy as np, glob
import sys

from src.models.sae import plSAE
from src.utils.process_utils import save_activations_to_npy, save_labels_to_npy
import numpy as np
from src.constants import CONFIG_PATH

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Dict[Any, Any]:
    """Training Script for SAE training [with black-box model inference prior]

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """

    if not HydraConfig.initialized():
        HydraConfig.instance().clear()
        HydraConfig().set_config(cfg)
    #hydra_run_dir = HydraConfig.get().run.dir
    hydra_run_dir = cfg.paths.output_dir
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger  = WandbLogger(
                project=cfg["CHECKPOINT"]["wandb_project"],
                entity=cfg["CHECKPOINT"]["wandb_user"],
                name=cfg["CHECKPOINT"]["experiment_name"],
            )

    try:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        log.warning(f"Could not push config to WandB: {e}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating SAE <{cfg.sae._target_}>")
    sae: plSAE = hydra.utils.instantiate(cfg.sae)

    log.info(f"Instantiating Backbone <{cfg.model.net._target_}>")
    net: nn.Module = hydra.utils.instantiate(cfg.model.net)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=[wandb_logger])

    callbacks_cfg = cfg.get("callbacks") or []
    callbacks: List[Callback] = [hydra.utils.instantiate(cb) for cb in callbacks_cfg]

    log.info(f"Instantiating trainer <{cfg.trainer_sae._target_}>")
    trainer_sae: Trainer = hydra.utils.instantiate(cfg.trainer_sae, callbacks=callbacks, logger=[wandb_logger])

    datamodule.setup()

    act_datamodule = NpyActDataModule(batch_size=cfg.sae_batch_size, train_npy_path=cfg.datamodule["train_path"], val_npy_path=cfg.datamodule["train_path"], test_npy_path=cfg.datamodule["train_path"])

    # Training the SAE
    trainer_sae.fit(model=sae, datamodule=act_datamodule, ckpt_path=cfg.get("sae_ckpt_path"))

    # Test the SAE
    trainer_sae.test(
    model=sae,
    datamodule=act_datamodule,
    ckpt_path="best"
)
    

class NpyActDataModule(LightningDataModule):
    def __init__(self, batch_size: int, train_npy_path: str,
                 val_npy_path: Optional[str] = None, test_npy_path: Optional[str] = None):
        super().__init__()
        self.batch_size = batch_size
        self.train_npy_path = train_npy_path
        self.val_npy_path = val_npy_path
        self.test_npy_path = test_npy_path


    def train_dataloader(self, num_workers: int = 8):
        dataset = NpyActivationDataset(self.train_npy_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, num_workers: int = 8):
        if self.val_npy_path is not None:
            dataset = NpyActivationDataset(self.val_npy_path)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            return []

    def test_dataloader(self, num_workers: int = 8):
        if self.test_npy_path is not None:
            dataset = NpyActivationDataset(self.test_npy_path)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            return []

class NpyActivationDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, mmap_mode='r')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())

@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="sae_config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()