# src/datamodules/activations_datamodule.py
from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence, Dict, Any
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule

class ActivationNpyDataset(Dataset):
    """
    Memory-mapped read-only Dataset for activations stored in a single .npy array.
    Expected shape: (N, F) or (N, C, H, W). For SAE, (N, F) is typical.
    Returns (x, x) by default for reconstruction objectives.
    """
    def __init__(
        self,
        npy_path: str,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        return_target: bool = True,
        dtype: str = "float32",
        mmap_mode: str = "r",
        flatten: bool = False,
        eps: float = 1e-6,
        index_slice: Optional[slice] = None,
    ):
        super().__init__()
        self.npy_path = npy_path
        self.arr = np.load(npy_path, mmap_mode=mmap_mode).astype(dtype, copy=False)
        self.return_target = return_target
        self.flatten = flatten
        self.eps = eps

        if index_slice is not None:
            self.arr = self.arr[index_slice]

        if self.flatten and self.arr.ndim > 2:
            self._feature_shape = (int(np.prod(self.arr.shape[1:])),)
        elif self.arr.ndim == 1:
            self.arr = self.arr[:, None]
            self._feature_shape = (1,)
        else:
            self._feature_shape = self.arr.shape[1:]

        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return self.arr.shape[0]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + self.eps)

    def __getitem__(self, idx: int):
        x = self.arr[idx]

        if self.flatten and x.ndim > 1:
            x = x.reshape(-1)

        x = self._normalize(x)

        x = torch.from_numpy(x)
        if self.return_target:
            return x, x 
        return x

class SimpleActivationsDataModule(LightningDataModule):
    """
    Minimal PL DataModule for SAE training from activations stored in .npy files.

    Modes:
      1) Provide train/val/test paths explicitly.
      2) Provide a single path and use split fractions.

    Normalization:
      - If normalize=True and no stats path is given, compute per-feature mean/std on TRAIN only.
      - Save to stats_path (.json or .npz) if provided.
      - If stats_path exists, load from it.
    """
    def __init__(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        single_path: Optional[str] = None,
        split_fracs: Tuple[float, float, float] = (0.9, 0.05, 0.05),
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        stats_path: Optional[str] = None,   # endswith .json or .npz
        flatten: bool = False,
        dtype: str = "float32",
        persistent_workers: bool = True,
        shuffle_train: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if single_path is not None:
            s = sum(split_fracs)
            assert abs(s - 1.0) < 1e-6, f"split_fracs must sum to 1.0, got {s}"

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self._mean = None
        self._std = None

        self._train_source_path = train_path or single_path

    def prepare_data(self) -> None:
        pass

    def _load_stats(self, feature_shape: Sequence[int]) -> None:
        sp = self.hparams.stats_path
        if sp is None or not os.path.exists(sp):
            return
        if sp.endswith(".npz"):
            data = np.load(sp)
            self._mean = data["mean"]
            self._std = data["std"]
        else:
            with open(sp, "r") as f:
                d = json.load(f)
            self._mean = np.asarray(d["mean"], dtype=self.hparams.dtype).reshape(feature_shape)
            self._std  = np.asarray(d["std"], dtype=self.hparams.dtype).reshape(feature_shape)

    def _save_stats(self) -> None:
        sp = self.hparams.stats_path
        if sp is None:
            return
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        if sp.endswith(".npz"):
            np.savez_compressed(sp, mean=self._mean, std=self._std)
        else:
            with open(sp, "w") as f:
                json.dump({"mean": self._mean.tolist(), "std": self._std.tolist()}, f)

    def _compute_train_stats(self, arr: np.ndarray, flatten: bool) -> Tuple[np.ndarray, np.ndarray]:
        if flatten and arr.ndim > 2:
            N = arr.shape[0]
            X = arr.reshape(N, -1)
            mean = np.asarray(X.mean(axis=0), dtype=self.hparams.dtype)
            std  = np.asarray(X.std(axis=0),  dtype=self.hparams.dtype)
            mean = mean.reshape((-1,))
            std  = std.reshape((-1,))
            return mean, std
        else:
            mean = np.asarray(arr.mean(axis=0), dtype=self.hparams.dtype)
            std  = np.asarray(arr.std(axis=0),  dtype=self.hparams.dtype)
            return mean, std

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        if hp.train_path or hp.val_path or hp.test_path:
            assert hp.train_path is not None, "train_path is required when using explicit files"
            train_arr = np.load(hp.train_path, mmap_mode="r").astype(hp.dtype, copy=False)

            dummy = ActivationNpyDataset(hp.train_path, flatten=hp.flatten, mmap_mode="r", dtype=hp.dtype)
            feature_shape = dummy._feature_shape
            self._load_stats(feature_shape)
            if hp.normalize and (self._mean is None or self._std is None):
                self._mean, self._std = self._compute_train_stats(train_arr, hp.flatten)
                self._save_stats()

            self._train_ds = ActivationNpyDataset(hp.train_path, self._mean, self._std,
                                                  return_target=True, flatten=hp.flatten, dtype=hp.dtype)
            self._val_ds = ActivationNpyDataset(hp.val_path, self._mean, self._std,
                                                return_target=True, flatten=hp.flatten, dtype=hp.dtype) if hp.val_path else None
            self._test_ds = ActivationNpyDataset(hp.test_path, self._mean, self._std,
                                                 return_target=True, flatten=hp.flatten, dtype=hp.dtype) if hp.test_path else None

        else:
            assert hp.single_path is not None, "Provide single_path or explicit train/val/test paths."
            arr = np.load(hp.single_path, mmap_mode="r").astype(hp.dtype, copy=False)
            N = arr.shape[0]
            n_train = int(N * hp.split_fracs[0])
            n_val   = int(N * hp.split_fracs[1])
            n_test  = N - n_train - n_val

            dummy = ActivationNpyDataset(hp.single_path, flatten=hp.flatten, mmap_mode="r", dtype=hp.dtype,
                                         index_slice=slice(0, n_train))
            feature_shape = dummy._feature_shape
            self._load_stats(feature_shape)
            if hp.normalize and (self._mean is None or self._std is None):
                self._mean, self._std = self._compute_train_stats(arr[:n_train], hp.flatten)
                self._save_stats()

            self._train_ds = ActivationNpyDataset(hp.single_path, self._mean, self._std, True,
                                                  flatten=hp.flatten, dtype=hp.dtype, index_slice=slice(0, n_train))
            self._val_ds   = ActivationNpyDataset(hp.single_path, self._mean, self._std, True,
                                                  flatten=hp.flatten, dtype=hp.dtype, index_slice=slice(n_train, n_train+n_val))
            self._test_ds  = ActivationNpyDataset(hp.single_path, self._mean, self._std, True,
                                                  flatten=hp.flatten, dtype=hp.dtype, index_slice=slice(n_train+n_val, N))

        torch.manual_seed(hp.seed)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle_train,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            return None
        return DataLoader(
            self._test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
