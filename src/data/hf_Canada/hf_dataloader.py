"""Build Dataloader for SITS from HuggingFace parquet"""

from __future__ import division, print_function

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.Canada.sampler import FileIDSampler, FWISampler

warnings.filterwarnings("ignore")

from src.constants import BANDS_10, BANDS_20, BANDS_60, BANDS_ALL, HF_IMG_COLUMNS


def get_dataloader(
    data_dir: str,
    transform: transforms.Compose = None,
    split: str = "train",
    bands: List[str] = BANDS_ALL,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    return_paths: bool = False,
    my_collate: Callable = None,
    target_file_id: Optional[int] = None,
    fwi_th: float = None,
    with_loc: bool = False,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Get Dataloader for SITS only.

    Args:
        data_dir (str): Directory of the downloaded parquet files
        transform (transforms.Compose, optional): Data augmentation pipeline. Defaults to None.
        split (str, optional): String indicating the split. Defaults to "train".
        bands (List[str], optional): List of bands to use. Defaults to BANDS_ALL.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for loading samples. Defaults to 4.
        shuffle (bool, optional): Flag to shuffle samples. Defaults to True.
        return_paths (bool, optional): Flag to return the sample info. Defaults to False.
        my_collate (Callable, optional): Collate function for the batch. Defaults to None.
        target_file_id (Optional[int], optional): File Id to sample only one specific sample type. Defaults to None.
        fwi_th (float, optional): FWI threshild for sampling specific fire risk. Defaults to None.
        with_loc (bool, optional): Flag to use localization. Defaults to False.

    Returns:
       torch.utils.data.DataLoader: SITS Dataloader
    """
    print("[DEBUG] entering get_dataloader", split)

    dataset = SatImDataset(
        data_dir=data_dir, transform=transform, split=split, return_paths=return_paths, bands=bands, with_loc=with_loc
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


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose = None,
        split: str = "train",
        return_paths: bool = False,
        bands: List[str] = BANDS_ALL,
        with_loc: bool = False,
    ):
        """Initialize the Dataset

        Args:
            data_dir (str): Directory of the downloaded parquet files
            transform (transforms.Compose, optional): Data augmentation pipeline. Defaults to None.
            split (str, optional): String indicating the split. Defaults to "train".
            return_paths (bool, optional): Flag to return the sample info. Defaults to False.
            bands (List[str], optional): List of bands to use. Defaults to BANDS_ALL.
            with_loc (bool, optional): Flag to use localization. Defaults to False.

        Raises:
            ValueError: Error in search split pattern
            NotImplementedError: Mixing bands across bands group is not possible
            NotImplementedError: Mixing bands across bands group is not possible
            NotImplementedError: Mixing bands across bands group is not possible
        """
        
        data_dir = Path(data_dir)
        split_patterns = {"train": "train-*.parquet", "val": "validation-*.parquet", "test": "test-*.parquet"}

        if split == "test_hard":
            # Load and filter test-* with file_id == "POS"

            print("Loading POS")
            test_files = sorted(str(p) for p in data_dir.rglob("test-*.parquet"))
            test_ds = HfDataset.from_parquet(test_files)

            print("Filtering POS")
            test_ds = test_ds.filter(
                lambda batch: [fid == "POS" for fid in batch["file_id"]], batched=True, batch_size=10
            )

            # Load and filter test_hard-* with file_id == "NEG"
            print("Loading NEG")
            test_hard_files = sorted(str(p) for p in data_dir.rglob("test_hard-*.parquet"))
            test_hard_ds = HfDataset.from_parquet(test_hard_files)

            print("Filtering Neg")
            test_hard_ds = test_hard_ds.filter(
                lambda batch: [fid == "NEG" for fid in batch["file_id"]], batched=True, batch_size=10
            )

            # Combine both
            combined_ds = concatenate_datasets([test_ds, test_hard_ds])
            self.hf_dataset = combined_ds.with_format(type="numpy", columns=HF_IMG_COLUMNS)

        else:
            pattern = split_patterns.get(split)
            if pattern is None:
                raise ValueError(f"Unsupported split: {split}")
            data_files = sorted(str(p) for p in data_dir.rglob(pattern))

            self.hf_dataset = HfDataset.from_parquet(data_files).with_format(type="numpy", columns=HF_IMG_COLUMNS)

        self.transform = transform
        self.return_paths = return_paths
        self.with_loc = with_loc

        # For __adapt__ method to respect the BANDS_10, BANDS_20, BANDS_60 order in the normalization
        self.bands = []
        if all([band in bands for band in BANDS_10]):
            self.band_10x = True
            self.bands.extend(BANDS_10)
        elif not any([band in bands for band in BANDS_10]):
            self.band_10x = False
        else:
            raise NotImplementedError("Mixing bands not supported yet")

        if all([band in bands for band in BANDS_20]):
            self.band_20x = True
            self.bands.extend(BANDS_20)
        elif not any([band in bands for band in BANDS_20]):
            self.band_20x = False
        else:
            raise NotImplementedError("Mixing bands not supported yet")

        if all([band in bands for band in BANDS_60]):
            self.band_60x = True
            self.bands.extend(BANDS_60)
        elif not any([band in bands for band in BANDS_60]):
            self.band_60x = False
        else:
            raise NotImplementedError("Mixing bands not supported yet")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):

        # If idx is a tensor, convert it to a list
        if torch.is_tensor(idx):
            idx = idx.item()

        sample, img_name_info = self.__fastadapt__(idx)

        if self.transform:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_name_info
        
        return sample

    def __fastadapt__(self, idx) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:

        # Extracting key information on S2 SITS
        item = self.hf_dataset[idx]

        # Equivalent to Img Name in PASTIS
        img_name_info = {"region": item["region"], "tile_id": item["tile_id"], "file_id": item["file_id"], "fwinx_mean": item["fwi"]}
        sample = {
            "10x": item["10x"] if self.band_10x else None,
            "20x": item["20x"] if self.band_20x else None,
            "60x": item["60x"] if self.band_60x else None,
            "labels": item["labels"],
            "doy": item["doy"],
        }

        if self.with_loc:
            sample["loc"] = item["loc"]

        return sample, img_name_info


def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b["unk_masks"].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
