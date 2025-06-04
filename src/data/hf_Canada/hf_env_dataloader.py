"""Build Dataloader for ENV only tiles from HuggingFace parquet"""

from __future__ import division, print_function

import json
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.Canada.sampler import FileIDSampler, FWISampler

warnings.filterwarnings("ignore")

from src.constants import HF_ENV_SPA_COLUMNS, HF_ENV_TAB_COLUMNS, TAB_SOURCE_COLS


def get_dataloader(
    data_dir: str,
    meta_hf: str,
    tab_transform: transforms.Compose = None,
    split: str = "train",
    tab_source_cols: List[str] = TAB_SOURCE_COLS,
    nan_value: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool =True,
    return_paths: bool = False,
    my_collate: Callable = None,
    target_file_id: int = None,
    fwi_th: float = None,
    with_loc: bool = False,
    is_spatial: bool = False,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Get Dataloader for ENV only.

    Args:
        data_dir (str): Directory of the downloaded parquet files
        meta_hf (str): Path to the metadata file containing source-cols mapping
        tab_transform (transforms.Compose, optional): Data augmentation pipeline. Defaults to None.
        split (str, optional): String indicating the split. Defaults to "train".
        tab_source_cols (List[str], optional): Dictionary mapping sources to target variables. Defaults to TAB_SOURCE_COLS.
        nan_value (float, optional): Value to fill missing entries. Defaults to 0.0.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for loading samples. Defaults to 4.
        shuffle (bool, optional): Flag to shuffle samples. Defaults to True.
        return_paths (bool, optional): Flag to return the sample info. Defaults to False.
        my_collate (Callable, optional): Collate function for the batch. Defaults to None.
        target_file_id (Optional[int], optional): File Id to sample only one specific sample type. Defaults to None.
        fwi_th (float, optional): FWI threshild for sampling specific fire risk. Defaults to None.
        with_loc (bool, optional): Flag to use localization. Defaults to False.
        is_spatial (bool, optional): Flag to use spatial inputs. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: ENV Dataloader
    """
    dataset = EnvDataset(
        data_dir=data_dir,
        meta_hf=meta_hf,
        tab_transform=tab_transform,
        split=split,
        return_paths=return_paths,
        tab_source_cols=tab_source_cols,
        nan_value=nan_value,
        with_loc=with_loc,
        is_spatial=is_spatial,
    )

    if target_file_id is not None:
        sampler = FileIDSampler(dataset, target_file_id, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=my_collate,
            sampler=sampler,
            prefetch_factor=2,
        )

    elif fwi_th is not None:
        sampler = FWISampler(dataset, fwi_th, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=my_collate,
            sampler=sampler,
            prefetch_factor=2,
        )

    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=my_collate,
            prefetch_factor=2,
        )

    return dataloader


def process_nan(tab_data: np.ndarray, nan_value: Union[str, float] = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Process NaN values in environmental data"""
    mask = np.isnan(tab_data)
    if isinstance(nan_value, float):
        tab_data = np.nan_to_num(tab_data, nan=nan_value)
    elif nan_value == "median":
        median_mod = np.nanmedian(tab_data, axis=0)
        tab_data = np.nan_to_num(tab_data, nan=median_mod)
        tab_data = np.nan_to_num(tab_data, nan=0.0)
    elif nan_value == "nan":
        pass
    else:
        raise ValueError("nan_value should be either 'median', 'nan' or a float")
    return tab_data, mask


class EnvDataset(Dataset):
    """ENV only dataset."""

    def __init__(
        self,
        data_dir: str,
        meta_hf: str,
        tab_transform: Optional[transforms.Compose] = None,
        split: str = "train",
        return_paths: bool = False,
        with_loc: bool = False,
        tab_source_cols: Dict[str, List[str]] = TAB_SOURCE_COLS,
        is_spatial: bool = False,
        nan_value: Union[str, float] = 0.0,
    ):
        """Initialize the Dataset

        Args:
            data_dir (str): Directory of the downloaded parquet files
            meta_hf (str): Path to the metadata file containing source-cols mapping
            tab_transform (Optional[transforms.Compose], optional): Data augmentation pipeline. Defaults to None.
            split (str, optional): String indicating the split. Defaults to "train".
            return_paths (bool, optional): Flag to return the sample info. Defaults to False.
            with_loc (bool, optional): Flag to use localization. Defaults to False.
            tab_source_cols (Dict[str, List[str]], optional): Dictionary mapping sources to target variables. Defaults to TAB_SOURCE_COLS.
            is_spatial (bool, optional): Flag to use spatial inputs. Defaults to False.
            nan_value (Union[str, float], optional): Value to fill missing entries. Defaults to 0.0.

        Raises:
            ValueError: Error in search split pattern
        """

        data_dir = Path(data_dir)
        split_patterns = {"train": "train-*.parquet", "val": "validation-*.parquet", "test": "test-*.parquet"}

        if split == "test_hard":
            test_files = sorted(str(p) for p in data_dir.rglob("test-*.parquet"))
            test_ds = HfDataset.from_parquet(test_files)
            test_ds = test_ds.filter(
                lambda batch: [fid == "POS" for fid in batch["file_id"]], batched=True, batch_size=10
            )

            test_hard_files = sorted(str(p) for p in data_dir.rglob("test_hard-*.parquet"))
            test_hard_ds = HfDataset.from_parquet(test_hard_files)
            test_hard_ds = test_hard_ds.filter(
                lambda batch: [fid == "NEG" for fid in batch["file_id"]], batched=True, batch_size=10
            )

            combined_ds = concatenate_datasets([test_ds, test_hard_ds])
            self.hf_dataset = combined_ds.with_format(
                type="numpy", columns=HF_ENV_SPA_COLUMNS if is_spatial else HF_ENV_TAB_COLUMNS
            )

        else:
            pattern = split_patterns.get(split)
            if pattern is None:
                raise ValueError(f"Unsupported split: {split}")
            data_files = sorted(str(p) for p in data_dir.rglob(pattern))

            self.hf_dataset = HfDataset.from_parquet(data_files).with_format(
                type="numpy", columns=HF_ENV_SPA_COLUMNS if is_spatial else HF_ENV_TAB_COLUMNS
            )

        with open(meta_hf, "r") as f:
            self.meta_hf = json.load(f)

        self.tab_transform = tab_transform
        self.return_paths = return_paths
        self.with_loc = with_loc
        self.tab_source_cols = tab_source_cols
        self.nan_value = nan_value
        self.is_spatial = is_spatial

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):

        # If idx is a tensor, convert it to a list
        if torch.is_tensor(idx):
            idx = idx.item()

        env_sample, img_name_info = self.__fastadapt__(idx)

        if self.tab_transform:
            env_sample = self.tab_transform(env_sample)

        if self.return_paths:
            return env_sample, img_name_info

        return env_sample

    def __fastadapt__(self, idx) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:

        # Extracting key information on S2 SITS
        item = self.hf_dataset[idx]

        # Equivalent to Img Name in PASTIS
        img_name_info = {"region": item["region"], "tile_id": item["tile_id"], "file_id": item["file_id"], "fwinx_mean": item["fwi"]}
        labels = item["labels"]

        if not self.is_spatial:
            # Loading the tabular data
            tab_ts = []
            tab_doy_ts = None

            for source, cols in self.tab_source_cols.items():
                tab_cols = self.meta_hf[f"tab_{source}"]
                cols_idx = [tab_cols.index(a) for a in sorted(cols)]  # Sort for consistency during normalization
                tab_data = item[f"tab_{source}"][:, cols_idx]
                tab_ts.append(tab_data)

            tab_doy_ts = item["env_doy"]

            tab_ts = np.concatenate(tab_ts, axis=1)  # T, C
            tab_ts, mask = process_nan(tab_ts, self.nan_value)
            env_sample = {"tab": tab_ts, "tab_doy": tab_doy_ts, "mask": mask, "labels": labels}

        else:
            env_sample = {}
            for source, cols in self.tab_source_cols.items():
                tab_cols = self.meta_hf[f"env_{source}"]
                cols_idx = [tab_cols.index(a) for a in sorted(cols)]
                tab_data = item[f"env_{source}"][cols_idx]  # Input: T, C, H, W on Hf directly
                # tab_data = np.moveaxis(tab_data, -1, 0)  # T, C, H, W (NOT Necessary)
                tab_data, tab_mask = process_nan(tab_data, self.nan_value)
                env_sample[source] = tab_data
                env_sample[f"{source}_mask"] = tab_mask

                if self.with_loc:
                    env_sample[f"{source}_loc"] = item[f"env_{source}_loc"]  # Input: H, W, 2

            tab_doy_ts = item["env_doy"]
            env_sample["tab_doy"] = tab_doy_ts
            env_sample["labels"] = labels

        if self.with_loc:
            raise NotImplementedError("Location not yet implemented")

        return env_sample, img_name_info
