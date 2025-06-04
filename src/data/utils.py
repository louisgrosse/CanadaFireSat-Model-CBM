"""Utils functions for Data Processing"""
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import rasterio
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.constants import BANDS_10, BANDS_20, BANDS_60


# Adapted from deepsat.data
def segmentation_ground_truths(sample: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = sample["labels"]
    if "unk_masks" in sample.keys():
        unk_masks = sample["unk_masks"]
    else:
        unk_masks = None

    if "edge_labels" in sample.keys():
        edge_labels = sample["edge_labels"]
        return labels, edge_labels, unk_masks
    return labels, unk_masks


# Keep the relative order of BANDS_10, BANDS_20, BANDS_60 for normalization: based on preprocessing and/or __adapt__ method
def extract_stats(stats_path: os.PathLike, bands: List[str]) -> np.ndarray:

    with open(stats_path, "r") as f:
        json_stats = json.load(f)

    stats_array_10x = (
        np.array([json_stats[band] for band in BANDS_10 if band in bands]).reshape(1, -1, 1, 1).astype(np.float32)
    )  # To be T * C * H * W
    stats_array_20x = (
        np.array([json_stats[band] for band in BANDS_20 if band in bands]).reshape(1, -1, 1, 1).astype(np.float32)
    )  # To be T * C * H * W
    stats_array_60x = (
        np.array([json_stats[band] for band in BANDS_60 if band in bands]).reshape(1, -1, 1, 1).astype(np.float32)
    )  # To be T * C * H * W

    stats_array = np.concatenate([stats_array_10x, stats_array_20x, stats_array_60x], axis=1)
    return stats_array
