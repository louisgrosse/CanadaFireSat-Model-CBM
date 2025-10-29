# This file is adapted from https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1/blob/main/ssl4eos12_dataset.py.


import os
import random
import torch
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

S2L1C_MEAN = [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712, 1021.753, 3205.112, 2545.798]
S2L1C_STD = [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]

S2L2A_MEAN = [1793.243, 1924.863, 2184.553, 2340.936, 2671.402, 3240.082, 3468.412, 3563.244, 3627.704, 3711.071, 3416.714, 2849.625]
S2L2A_STD = [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307]

S1GRD_MEAN = [-12.577, -20.265]
S1GRD_STD = [5.179, 5.872]

S2RGB_MEAN = [100.708, 87.489, 61.932]
S2RGB_STD = [68.550, 47.647, 40.592]


class SSL4EOS12Dataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
            split_file: Path = None,
            modalities: list = None,
            transform: transforms.Compose = None,
            concat: bool = False,
            single_timestamp: bool = False,
            num_timestamps: int = 4,
            num_batch_samples: int = None,
            caption_col: str = 'caption',
    ):
        """
        Dataset class for the SSL4EOS12 V1.1 dataset.
        :param data_dir: Path to data directory of the selected split.
        :param split_file: optional, txt file with list of zarr.zip file name. Reduces initialization time.
        :param modalities: list of modalities folders, defaults to ['S2L1C', 'S2L2A', 'S1GRD'].
        :param transform: tranform function that processes a dict or numpy array (if concat=True).
        :param concat: Concatenate all modalities along the band dimension.
        :param single_timestamp: Loads a single timestamp instead of all four timestamps.
        :param num_batch_samples: Subsample samples in zarr files, e.g. if GPU memory is not sufficient.
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ['S2L1C', 'S2L2A', 'S1GRD', 'captions']
        self.transform = transform
        self.concat = concat
        self.num_batch_samples = num_batch_samples
        self.caption_col = caption_col

        # Get sample names
        if split_file is not None:
            with open(split_file, 'r') as f:
                self.samples = f.read().splitlines()
            # Remove file extension
            self.samples = [f.split('.', 1)[0] for f in self.samples]
        else:
            self.samples = os.listdir(self.data_dir / self.modalities[0])
            self.samples = [f.split('.', 1)[0] for f in self.samples
                            if f.endswith('.zarr.zip') or f.endswith('.csv')]
            assert len(self.samples) > 0, "No samples found."

        # Get file extension for each modality
        self.file_extentions = {}
        for modality in self.modalities:
            files = list(self.data_dir.glob(f"{modality}/{self.samples[0]}*"))
            assert len(files) > 0, f"No samples found for modality {modality} in {self.data_dir}/{modality}."
            self.file_extentions[modality] = files[0].name.split('.', 1)[1]

        self.single_timestamp = single_timestamp
        self.num_timestamps = num_timestamps
        if single_timestamp:
            # Repeat samples to include all timestamps in the dataset
            self.samples = np.repeat(self.samples, num_timestamps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx: Index of zarr.zip file.
        :return: dict of modalities or tensor (if concat=True) with dims [B, T, C, H, W] or [B, C, H, W]
            (if single_timestamp=True).
        """
        data = {}
        # Load numpy values for each modality from zarr.zip files
        for modality in self.modalities:
            if self.file_extentions[modality] == "zarr.zip":
                ds = xr.open_zarr((self.data_dir / modality / self.samples[idx]).with_suffix(".zarr.zip"))
                if self.single_timestamp:
                    # Select a single timestamp
                    ds = ds.isel(time=idx % self.num_timestamps)
                data[modality] = ds.bands.values
            elif self.file_extentions[modality] == "csv":
                df = pd.read_csv((self.data_dir / modality / self.samples[idx]).with_suffix(".csv"), index_col=0)
                data[modality] = df[self.caption_col].values.reshape(64, 4)
                if self.single_timestamp:
                    # Select a single timestamp
                    data[modality] = data[modality][:, idx % self.num_timestamps]
            else:
                raise NotImplementedError(f"File extention {self.file_extentions[modality]} not supported.")

        num_samples = data[self.modalities[0]].shape[0]
        if self.num_batch_samples is not None and self.num_batch_samples != num_samples:
            # Subsample samples
            selected = random.sample(list(range(num_samples)), k=self.num_batch_samples)
            for modality in self.modalities:
                data[modality] = data[modality][selected]

        # Pop txt data
        txt_data = {
            modality: data.pop(modality) for modality in self.modalities if self.file_extentions[modality] == "csv"
        }

        # Save band dims in case of dict outputs
        num_band_dims = {m: data[m].shape[-3] for m in data.keys()}
        band_dims_idx = {m: n for m, n in zip(data.keys(), [0] + np.cumsum(list(num_band_dims.values())).tolist())}

        # Concatenate along band dim for transform and convert to Tensor
        data = torch.Tensor(np.concatenate(list(data.values()), axis=-3))

        if self.transform is not None:
            data = self.transform(data)

        if not self.concat:
            # Split up modality data and return as dict
            data = {m: data[..., band_dims_idx[m]:band_dims_idx[m]+num_band_dims[m], :, :]
                    for m in num_band_dims.keys()}
            for modality, value in txt_data.items():
                # Join txt data
                data[modality] = value
        else:
            assert len(txt_data) < 2, "Current code expects maximum one text modality if concat=True."
            data = {
                "image": data,
                "caption": list(txt_data.values())[0],
            }

        return data


def collate_fn(batch):
    if isinstance(batch, dict) or isinstance(batch, torch.Tensor):
        # Single sample
        return batch
    elif isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        # Concatenate tensors along sample dim
        return torch.concat(batch, dim=0)
    elif isinstance(batch, list) and isinstance(batch[0], np.ndarray):
        # Concatenate arrays along sample dim
        return np.concatenate(batch, axis=0)
    elif isinstance(batch, list) and isinstance(batch[0], dict):
        # Concatenate each modality tensor along sample dim
        return {
            m: collate_fn([b[m] for b in batch])
            for m in batch[0].keys()
        }
