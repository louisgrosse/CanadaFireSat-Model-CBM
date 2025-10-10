"""Library of lightning datamodules for different data type from Hf Datasets"""
from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule

from src.constants import ENV_SOURCE_COLS, TAB_SOURCE_COLS
from src.data.augmentations import MixHVFlip
from src.data.Canada.data_transforms import (
    Canada_segmentation_transform,
    EnvCanada_segmentation_transform,
    TabCanada_segmentation_transform,
)
from src.data.hf_Canada.hf_dataloader import get_dataloader
from src.data.hf_Canada.hf_env_dataloader import get_dataloader as get_env_dataloader
from src.data.hf_Canada.hf_mix_dataloader import get_dataloader as get_tab_dataloader

import os
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import pyarrow.parquet as pq
import glob

from src.data.utils import extract_stats
from src.data.augmentations import ToTensor, Normalize, Rescale, Concat

class WorldStratPairDataset(Dataset):
    """
    Memory-efficient streaming dataset that loads one parquet shard at a time.
    Each __getitem__ dynamically accesses rows without keeping all shards in RAM.
    """

    def __init__(self, parquet_dir, split, mean_file, std_file, bands, is_training=True):
        super().__init__()
        self.parquet_files = sorted(glob.glob(os.path.join(parquet_dir, f"{split}_*.parquet")))
        if len(self.parquet_files) == 0:
            raise FileNotFoundError(f"No parquet shards found in {parquet_dir} for pattern {split}_*.parquet")

        # Compute cumulative row offsets so we can index globally
        self.tables_meta = []
        self.total_rows = 0
        for f in self.parquet_files:
            meta = pq.read_metadata(f)
            nrows = meta.num_rows
            self.tables_meta.append((f, self.total_rows, self.total_rows + nrows))
            self.total_rows += nrows

        print(f"[WorldStratPairDataset] {len(self.parquet_files)} shards, {self.total_rows} total rows")

        # Normalization / transforms
        mean_array = extract_stats(mean_file, bands)
        std_array = extract_stats(std_file, bands)
        self.transform = transforms.Compose([
            ToTensor(with_loc=False),
            Rescale(output_size=(224, 224)),
            Concat(concat_keys=["10x", "20x", "60x"]),
            Normalize(mean=mean_array, std=std_array),
        ])

        self.is_training = is_training

    def __len__(self):
        return self.total_rows

    def _load_row(self, idx):
        # find which shard contains this row
        for path, start, end in self.tables_meta:
            if start <= idx < end:
                local_idx = idx - start
                table = pq.read_table(path, columns=[
                    "10x_L1C","20x_L1C","60x_L1C",
                    "10x_L2A","20x_L2A","60x_L2A"
                ], use_threads=False)
                df = table.to_pandas()
                row = df.iloc[local_idx]
                return row
        raise IndexError

    def __getitem__(self, idx):
        row = self._load_row(idx)

        def _to_np(prefix):
            return {
                "10x": np.array(row[f"10x_{prefix}"], dtype=np.float32),
                "20x": np.array(row[f"20x_{prefix}"], dtype=np.float32),
                "60x": np.array(row[f"60x_{prefix}"], dtype=np.float32),
            }

        sample_L1C = self.transform(_to_np("L1C"))
        sample_L2A = self.transform(_to_np("L2A"))
        return {"l1c": sample_L1C, "l2a": sample_L2A}
    
    
class SatWorldStratDataModule(LightningDataModule):
    """LightningDataModule for paired WorldStrat dataset used to train the L1C→L2A adapter."""

    def __init__(self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs):
        super().__init__()
        self.batch_size = train_config["batch_size"]
        self.num_workers = train_config["num_workers"]
        self.data_dir = train_config["data_dir"]

        self.mean_file = kwargs["mean_file"]
        self.std_file = kwargs["std_file"]
        self.bands = kwargs["bands"]

    def setup(self, stage=None):
        # Nothing special needed since each split loads its own parquets
        pass

    def train_dataloader(self):
        ds = WorldStratPairDataset(
            parquet_dir=self.data_dir,
            split="worldstrat_train",
            mean_file=self.mean_file,
            std_file=self.std_file,
            bands=self.bands,
            is_training=True,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        ds = WorldStratPairDataset(
            parquet_dir=self.data_dir,
            split="worldstrat_val",
            mean_file=self.mean_file,
            std_file=self.std_file,
            bands=self.bands,
            is_training=False,
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        ds = WorldStratPairDataset(
            parquet_dir=self.data_dir,
            split="worldstrat_test",
            mean_file=self.mean_file,
            std_file=self.std_file,
            bands=self.bands,
            is_training=False,
        )
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


class SatDataModule(LightningDataModule):
    """Datamodule for the SITS"""
    def __init__(
        self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs
    ):
        super().__init__()

        # Train key variables
        self.train_batch_size = train_config["batch_size"]
        self.train_num_workers = train_config["num_workers"]
        self.train_data_dir = train_config["data_dir"]
        self.train_transform = Canada_segmentation_transform(model_config=model_config, is_training=True, **kwargs)
        self.target_file_id = train_config["target_file_id"] if "target_file_id" in train_config else None
        self.fwi_th = (
            train_config["fwi_ths"][0] if ("fwi_ths" in train_config) and (0 in train_config["fwi_ths"]) else None
        )

        # Validation key variables
        self.val_batch_size = eval_config["batch_size"]
        self.val_num_workers = eval_config["num_workers"]
        self.val_data_dir = eval_config["data_dir"]
        self.val_transform = Canada_segmentation_transform(model_config=model_config, is_training=False, **kwargs)

        self.kwargs = kwargs
        self.test_transform = Canada_segmentation_transform(
            model_config=model_config, is_training=False, is_eval=True, **kwargs
        )

        # --- Sanity check: verify MS-CLIP normalization ---
        #print("\n=== Sanity check: MS-CLIP normalization ===")
        
        #tmp_loader = self.train_dataloader()  # creates a temporary dataloader
        #batch = next(iter(tmp_loader))
        #x = batch["inputs"] if "inputs" in batch else batch[list(batch.keys())[0]]

        #print(f"Shape: {x.shape}")
        #print(f"Min: {x.min().item():.3f}, Max: {x.max().item():.3f}")
        #print(f"Mean: {x.mean().item():.3f}, Std: {x.std().item():.3f}")
        #print("Per-channel mean:", x.mean(dim=(0,2,3)))
        #print("Per-channel std:", x.std(dim=(0,2,3)))
        #print("===========================================\n")


        # call once on a batch
        #msclip_band_order_checks(x)


    def __len__(self):
        return len(self.train_dataloader().dataset)


    def train_dataloader(self, target_file_id: Optional[int] = None, fwi_th: Optional[float] = None):

        if target_file_id is not None:
            _target_file_id = target_file_id
        else:
            _target_file_id = self.target_file_id

        if fwi_th is not None:
            _fwi_th = fwi_th
        else:
            _fwi_th = self.fwi_th

        return get_dataloader(
            data_dir=self.train_data_dir,
            split="train",
            transform=self.train_transform,
            batch_size=self.train_batch_size,
            fwi_th=_fwi_th,
            target_file_id=_target_file_id,
            num_workers=self.train_num_workers,
            **self.kwargs,
        )

    def val_dataloader(self):
        return get_dataloader(
            data_dir=self.val_data_dir,
            split="val",
            transform=self.val_transform,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            **self.kwargs,
        )

    def test_dataloader(self, split: str = "test"):
        # Files path should be the same for test and val
        return get_dataloader(
            data_dir=self.val_data_dir,
            split=split,
            transform=self.test_transform,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            return_paths=True,
            **self.kwargs,
        )


class TabSatDataModule(LightningDataModule):
    """Datamodule for Mix data"""
    def __init__(
        self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs
    ):
        super().__init__()

        # Train key variables
        self.train_data_dir = train_config["data_dir"]
        self.train_batch_size = train_config["batch_size"]
        self.train_num_workers = train_config["num_workers"]
        if "is_spatial" in kwargs and kwargs["is_spatial"]:
            self.train_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=True, img_only=False, **kwargs
            )
            self.tab_train_transform = EnvCanada_segmentation_transform(
                model_config=model_config, is_training=True, env_only=False, **kwargs
            )
            self.mix_train_transform = MixHVFlip(hflip_prob=0.5, vflip_prob=0.5, with_loc=kwargs["with_loc"])
            self.tab_train_cols = ENV_SOURCE_COLS
        else:
            self.train_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=True, img_only=True, **kwargs
            )
            self.tab_train_transform = TabCanada_segmentation_transform(
                model_config=model_config, is_training=True, **kwargs
            )
            self.mix_train_transform = None
            self.tab_train_cols = TAB_SOURCE_COLS
        self.target_file_id = train_config["target_file_id"] if "target_file_id" in train_config else None
        self.fwi_th = (
            train_config["fwi_ths"][0] if ("fwi_ths" in train_config) and (0 in train_config["fwi_ths"]) else None
        )

        # Eval key variables
        self.val_data_dir = eval_config["data_dir"]
        self.val_batch_size = eval_config["batch_size"]
        self.val_num_workers = eval_config["num_workers"]
        if "is_spatial" in kwargs and kwargs["is_spatial"]:
            self.val_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=False, img_only=False, **kwargs
            )
            self.tab_val_transform = EnvCanada_segmentation_transform(
                model_config=model_config, is_training=False, env_only=False, **kwargs
            )

            self.tab_val_cols = ENV_SOURCE_COLS
        else:
            self.val_transform = Canada_segmentation_transform(
                model_config=model_config, is_training=False, img_only=True, **kwargs
            )
            self.tab_val_transform = TabCanada_segmentation_transform(
                model_config=model_config, is_training=False, **kwargs
            )
            self.tab_val_cols = TAB_SOURCE_COLS
        self.kwargs = kwargs
        self.test_transform = Canada_segmentation_transform(
            model_config=model_config, is_training=False, is_eval=True, **kwargs
        )

    def __len__(self):
        return len(self.train_dataloader().dataset)


    def train_dataloader(self, target_file_id: Optional[int] = None, fwi_th: Optional[float] = None):

        if target_file_id is not None:
            _target_file_id = target_file_id
        else:
            _target_file_id = self.target_file_id

        if fwi_th is not None:
            _fwi_th = fwi_th
        else:
            _fwi_th = self.fwi_th

        return get_tab_dataloader(
            data_dir=self.train_data_dir,
            split="train",
            transform=self.train_transform,
            tab_transform=self.tab_train_transform,
            mix_transform=self.mix_train_transform,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            target_file_id=_target_file_id,
            fwi_th=_fwi_th,
            tab_source_cols=self.tab_train_cols,
            **self.kwargs,
        )

    def val_dataloader(self):
        return get_tab_dataloader(
            data_dir=self.val_data_dir,
            split="val",
            transform=self.val_transform,
            tab_transform=self.tab_val_transform,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )

    def test_dataloader(self, split: str = "test"):
        return get_tab_dataloader(
            data_dir=self.val_data_dir,
            split=split,
            transform=self.test_transform,
            tab_transform=self.tab_val_transform,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            return_paths=True,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )


class EnvDataModule(LightningDataModule):
    """Datamodule for ENV only data"""
    def __init__(
        self, model_config: Dict[str, Any], train_config: Dict[str, Any], eval_config: Dict[str, Any], **kwargs
    ):
        super().__init__()

        # Train key variables
        self.train_data_dir = train_config["data_dir"]
        self.train_batch_size = train_config["batch_size"]
        self.train_num_workers = train_config["num_workers"]
        self.tab_train_transform = EnvCanada_segmentation_transform(model_config=model_config, env_only=True, **kwargs)
        self.tab_train_cols = ENV_SOURCE_COLS
        self.target_file_id = train_config["target_file_id"] if "target_file_id" in train_config else None
        self.fwi_th = (
            train_config["fwi_ths"][0] if ("fwi_ths" in train_config) and (0 in train_config["fwi_ths"]) else None
        )

        # Eval key variables
        self.val_data_dir = eval_config["data_dir"]
        self.val_batch_size = eval_config["batch_size"]
        self.val_num_workers = eval_config["num_workers"]
        self.tab_val_transform = EnvCanada_segmentation_transform(
            model_config=model_config, env_only=True, is_training=False, **kwargs
        )
        self.tab_val_cols = ENV_SOURCE_COLS
        self.kwargs = kwargs

    def __len__(self):
        return len(self.train_dataloader().dataset)


    def train_dataloader(self, target_file_id: Optional[int] = None, fwi_th: Optional[float] = None):

        if target_file_id is not None:
            _target_file_id = target_file_id
        else:
            _target_file_id = self.target_file_id

        if fwi_th is not None:
            _fwi_th = fwi_th
        else:
            _fwi_th = self.fwi_th

        return get_env_dataloader(
            data_dir=self.train_data_dir,
            split="train",
            tab_transform=self.tab_train_transform,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            target_file_id=_target_file_id,
            fwi_th=_fwi_th,
            tab_source_cols=self.tab_train_cols,
            **self.kwargs,
        )

    def val_dataloader(self):
        return get_env_dataloader(
            data_dir=self.val_data_dir,
            split="val",
            tab_transform=self.tab_val_transform,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )

    def test_dataloader(self, split: str = "test"):
        return get_env_dataloader(
            data_dir=self.val_data_dir,
            split=split,
            tab_transform=self.tab_val_transform,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            return_paths=True,
            tab_source_cols=self.tab_val_cols,
            **self.kwargs,
        )


def msclip_band_order_checks(x):  # x: [B,T,C,H,W] after MS-CLIP normalization
    B,T,C,H,W = x.shape
    # Undo normalization just for these checks (optional but more interpretable)
    # If you kept the normalized tensor, skip “denorm” and operate on x directly.
    # Using your MSCLIP_MEANS/STDS on device:
    means = torch.tensor([925.161,1183.128,1338.041,1667.254,2233.633,2460.96,2555.569,2619.542,2406.497,1841.645], device=x.device).view(1,1,C,1,1)
    stds  = torch.tensor([1205.586,1223.713,1399.638,1403.298,1378.513,1434.924,1491.141,1454.089,1473.248,1365.08], device=x.device).view(1,1,C,1,1)
    x_reflect = x * stds + means  # back to reflectance-like scale

    # Collapse B,T,H,W
    xr = x_reflect.reshape(-1, C, H, W)

    # 1) NDVI sanity: (B8 - B4) / (B8 + B4) should be mostly in [-1,1], skew > 0 for vegetated scenes
    B4 = xr[:, 2]   # index 2 if order is B2,B3,B4,...
    B8 = xr[:, 6]   # index 6
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    print("NDVI mean/std/min/max:", ndvi.mean().item(), ndvi.std().item(), ndvi.min().item(), ndvi.max().item())

    # Heuristic: global NDVI mean around > 0.1 on average scenes is reasonable; if it’s strongly negative, bands may be swapped.

    # 2) B8A should be highly correlated with B8 (same NIR neighborhood)
    B8A = xr[:, 7]
    def corr2(a,b):
        am = a.mean(dim=(1,2), keepdim=True); bm = b.mean(dim=(1,2), keepdim=True)
        an = (a-am); bn = (b-bm)
        num = (an*bn).mean(dim=(1,2))
        den = (an.pow(2).mean(dim=(1,2)).sqrt() * bn.pow(2).mean(dim=(1,2)).sqrt() + 1e-6)
        return (num/den).mean().item()
    print("corr(B8,B8A):", corr2(B8,B8A))

    # 3) SWIR (B11,B12) should correlate strongly with each other and less with Blue (B2)
    B2  = xr[:, 0]
    B11 = xr[:, 8]; B12 = xr[:, 9]
    print("corr(B11,B12):", corr2(B11,B12))
    print("corr(B2,B11):", corr2(B2,B11), "corr(B2,B12):", corr2(B2,B12))

    # 4) Mean brightness ordering heuristic (weak but quick):
    # Typically (dataset dependent): mean(B8) > mean(B4) > mean(B3) > mean(B2)
    ch_means = xr.mean(dim=(0,2,3))
    print("Per-channel reflectance means (B2..B12):", ch_means.tolist())