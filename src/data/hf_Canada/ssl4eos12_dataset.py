
import os
import random
import torch
import xarray as xr
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pytorch_lightning import LightningDataModule


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
            data_dir: str = None,
            split_file: str = None,
            modalities: list = None,
            transform: transforms.Compose = None,
            concat: bool = False,
            single_timestamp: bool = False,
            num_timestamps: int = 4,
            num_batch_samples: int = None,
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
        self.modalities = modalities or ['S2L1C', 'S2L2A', 'S1GRD']
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=S2L1C_MEAN,
                    std=S2L1C_STD,
                ),
            ])
        self.concat = concat
        self.num_batch_samples = num_batch_samples

        if split_file is not None:
            with open(split_file, 'r') as f:
                self.samples = f.read().splitlines()
        else:
            self.samples = os.listdir(self.data_dir / self.modalities[0])
            self.samples = [f for f in self.samples if f.endswith('.zarr.zip')]

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
            ds = xr.open_zarr(self.data_dir / modality / self.samples[idx])
            if self.single_timestamp:
                # Select a single timestamp
                ds = ds.isel(time=idx % self.num_timestamps)
            data[modality] = ds.bands.values

        num_samples = data[self.modalities[0]].shape[0]
        if self.num_batch_samples is not None and self.num_batch_samples != num_samples:
            # Subsample samples
            selected = random.sample(list(range(num_samples)), k=self.num_batch_samples)
            for modality in self.modalities:
                data[modality] = data[modality][selected]

        # Save band dims in case of dict outputs
        num_band_dims = {m: data[m].shape[-3] for m in self.modalities}
        band_dims_idx = {m: n for m, n in zip(self.modalities, [0] + np.cumsum(list(num_band_dims.values())).tolist())}

        # Concatenate along band dim for transform and convert to Tensor
        data = torch.Tensor(np.concatenate(list(data.values()), axis=-3))

        if self.transform is not None:
            data = self.transform(data)

        if not self.concat:
            # Split up modality data and return as dict
            data = {m: data[..., band_dims_idx[m]:band_dims_idx[m]+num_band_dims[m], :, :]
                    for m in self.modalities}

        return data


def collate_fn(batch):
    if isinstance(batch, dict) or isinstance(batch, torch.Tensor):
        # Single sample
        return batch
    elif isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        # Concatenate tensors along sample dim
        return {"inputs":torch.concat(batch, dim=0)[:, [1,2,3,4,5,6,7,8,11,12], :, :]}
    elif isinstance(batch, list) and isinstance(batch[0], dict):
        # Special case: using only S2L1C and feeding SAE
        if "S2L1C" in batch[0] and len(batch[0]) == 1:
            x = torch.concat([b["S2L1C"] for b in batch], dim=0)[:, [1,2,3,4,5,6,7,8,11,12], :, :]
            return {"inputs": x}
        # Fallback: keep previous behavior for multi-modal use
        return {
            m: torch.concat([b[m] for b in batch], dim=0)
            for m in batch[0].keys()
        }


class SSL4EOS12DataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        test_dir: str = None,
        train_split_file: str = None,
        val_split_file: str = None,
        test_split_file: str = None,
        modalities=None,
        concat: bool = True,
        single_timestamp: bool = True,
        num_timestamps: int = 4,
        num_batch_samples: int = None,
        batch_size: int = 4,
        num_workers: int = 8,
        max_train_samples: int = None,   # <-- use this to cap at ~177k
        val_fraction: float = 0.1,       # random val split if no explicit val split
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["modalities"])
        self.modalities = modalities

    def setup(self, stage=None):
        # ---- TRAIN ----
        self.train_dataset = SSL4EOS12Dataset(
            data_dir=self.hparams.train_dir,
            split_file=self.hparams.train_split_file,
            modalities=self.modalities,
            transform=None,  # keep dataset's own default transform
            concat=self.hparams.concat,
            single_timestamp=self.hparams.single_timestamp,
            num_timestamps=self.hparams.num_timestamps,
            num_batch_samples=self.hparams.num_batch_samples,
        )

        # cap number of train samples to ~177k (such that we get results comparable to the one with CanadaFireSat)
        if (
            self.hparams.max_train_samples is not None
            and len(self.train_dataset) > self.hparams.max_train_samples
        ):
            import torch
            idx = torch.randperm(len(self.train_dataset))[: self.hparams.max_train_samples].tolist()
            self.train_dataset = Subset(self.train_dataset, idx)

        # ---- VAL ----
        if self.hparams.val_dir is not None or self.hparams.val_split_file is not None:
            # Use explicit validation split (dir or split file)
            val_dir = self.hparams.val_dir or self.hparams.train_dir
            self.val_dataset = SSL4EOS12Dataset(
                data_dir=val_dir,
                split_file=self.hparams.val_split_file,
                modalities=self.modalities,
                transform=None,
                concat=self.hparams.concat,
                single_timestamp=self.hparams.single_timestamp,
                num_timestamps=self.hparams.num_timestamps,
                num_batch_samples=self.hparams.num_batch_samples,
            )
        elif self.hparams.val_fraction > 0:
            # Random train/val split from the (possibly sub-sampled) train set
            full_len = len(self.train_dataset)
            val_len = int(full_len * self.hparams.val_fraction)
            train_len = full_len - val_len
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [train_len, val_len]
            )
        else:
            self.val_dataset = None

        # ---- TEST ----
        if self.hparams.test_dir is not None or self.hparams.test_split_file is not None:
            test_dir = self.hparams.test_dir or self.hparams.train_dir
            self.test_dataset = SSL4EOS12Dataset(
                data_dir=test_dir,
                split_file=self.hparams.test_split_file,
                modalities=self.modalities,
                transform=None,
                concat=self.hparams.concat,
                single_timestamp=self.hparams.single_timestamp,
                num_timestamps=self.hparams.num_timestamps,
                num_batch_samples=self.hparams.num_batch_samples,
            )
        else:
            self.test_dataset = None

    # ---- LOADERS ----
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
