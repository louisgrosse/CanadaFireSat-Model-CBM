"""Script for baseline based on FWI"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
import rasterio
import torch
from omegaconf import DictConfig, ListConfig
from rasterio.windows import Window, from_bounds
from torch.nn import functional as F
from tqdm import tqdm

from deepsat.metrics.numpy_metrics import get_classification_metrics
from src.constants import (
    CANADA_REF_COLS,
    CONFIG_PATH,
    LABEL_RES,
    TEST_FILTER,
    TRAIN_FILTER,
    VAL_FILTER,
)
from src.eval.utils import get_pr_auc_scores


def _temp_grid(start_year: int, end_year: int, offset: int) -> List[datetime]:
    dates = []

    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        current_date = start_date

        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=offset)

    return dates


def _downsample_lab(label: np.ndarray, out_H: int, out_W: int) -> np.ndarray:
    label = torch.Tensor(label)
    kernel_size = (label.shape[0] // out_H, label.shape[1] // out_W)
    label = F.max_pool2d(label.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size).squeeze(0).squeeze(0)
    label = label.numpy()
    return label


def fast_resize_with_nan(data, scale_y, scale_x):
    """Process NaN value for interpolation"""

    # Fill NaNs with local mean
    median = np.nanmedian(data)
    data_filled = np.nan_to_num(data, nan=median)

    # Resize the data
    new_size = (int(data.shape[1] * scale_x), int(data.shape[0] * scale_y))
    data_resized = cv2.resize(data_filled, new_size, interpolation=cv2.INTER_LINEAR)

    # Create a mask of the NaN values after resizing
    nan_mask = np.isnan(cv2.resize(data, new_size, interpolation=cv2.INTER_NEAREST))

    # Restore NaNs in the resized data
    data_resized[nan_mask] = np.nan

    return data_resized


def load_fwi_tile(fire_date: datetime.date, horizon_type: str, fwi_file_path: os.PathLike):
    """Load the proper tile based on the evaluation set-up"""

    if horizon_type == "8-day Prediction":
        fwi_date = fire_date - timedelta(days=8)
        fwi_date_str = fwi_date.strftime("%Y%m%d")
        fwi_year = fwi_date.year
        if fwi_date.year == fire_date.year:
            fwi_file = os.path.join(fwi_file_path, f"{fwi_year}", f"fwi_dc_agg_{fwi_date_str}.tif")
        else:
            dates = _temp_grid(fwi_year, fwi_year, 8)
            closest_date = min(dates, key=lambda x: abs(x - fwi_date))
            fwi_date_str = closest_date.strftime("%Y%m%d")
            fwi_file = os.path.join(fwi_file_path, f"{fwi_year}", f"fwi_dc_agg_{fwi_date_str}.tif")

    elif horizon_type == "8-day Current":
        fwi_date = fire_date
        fwi_date_str = fwi_date.strftime("%Y%m%d")
        fwi_year = fwi_date.year
        fwi_file = os.path.join(fwi_file_path, f"{fwi_year}", f"fwi_dc_agg_{fwi_date_str}.tif")

    else:
        raise NotImplementedError(f"Invalid horizon type: {horizon_type}")

    return fwi_file


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="fwi_baseline")
def compute_fwi_baseline(cfg: DictConfig):
    """Evaluation Function"""

    # Loading split file
    dataset = pd.read_json(cfg.json_file, lines=True)[CANADA_REF_COLS + ["date"]]
    if cfg.split == "train":
        for column, value in TRAIN_FILTER.items():
            if len(value) > 0:
                dataset = dataset[dataset[column].isin(value)].reset_index(drop=True)
    elif cfg.split == "val":
        for column, value in VAL_FILTER.items():
            if len(value) > 0:
                dataset = dataset[dataset[column].isin(value)].reset_index(drop=True)
    else:
        for column, value in TEST_FILTER.items():
            if len(value) > 0:
                dataset = dataset[dataset[column].isin(value)].reset_index(drop=True)

    if isinstance(cfg.root_dir, str):
        root_dir = {0: cfg.root_dir}
    elif isinstance(cfg.root_dir, (list, ListConfig)):
        root_dir = {file_id: root_dir_ for file_id, root_dir_ in enumerate(cfg.root_dir)}

    predictions = [[] for _ in range(len(cfg.fwi_thresholds))]
    tot_probs = []
    labels = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="FWI Baseline Evaluation"):

        sorted_tile_names = sorted(row["valid_names"], key=lambda x: datetime.strptime(x.split("T")[0], "%Y%m%d"))
        ref_tile = os.path.join(
            root_dir[row["file_id"]], row["region"], str(row["tile_id"]), sorted_tile_names[-1], "B2.tif"
        )
        fire_date = row["date"]

        if row["file_id"] == 0:
            label = np.load(os.path.join(cfg.label_dir, str(row["tile_id"]), "label.npy"))
            label = _downsample_lab(label, LABEL_RES, LABEL_RES)
        else:
            label = np.zeros((LABEL_RES, LABEL_RES))

        labels.append(label.reshape(-1))

        with rasterio.open(ref_tile) as src:
            ref_bounds = src.bounds
            transform = src.transform
            center_x = (ref_bounds.left + ref_bounds.right) / 2
            center_y = (ref_bounds.top + ref_bounds.bottom) / 2
            pixel_width, pixel_height = transform[0], abs(transform[4])

        interpolation_x = (cfg.fwi_res / pixel_width) * cfg.label_res
        interpolation_y = (cfg.fwi_res / pixel_height) * cfg.label_res
        fwi_tile = load_fwi_tile(fire_date, cfg.horizon_type, cfg.fwi_file_path)

        with rasterio.open(fwi_tile) as src:
            transform = src.transform

            if cfg.window_type == "pixel":
                center_row, center_col = src.index(center_x, center_y)
                col_off = center_col - cfg.window_size // 2
                row_off = center_row - cfg.window_size // 2
                window = Window(col_off, row_off, cfg.window_size, cfg.window_size)
                data = src.read(2, window=window)  # Second band is the FWI mean
                window_transform = src.window_transform(window)

            elif cfg.window_type == "bounds":
                left = center_x - cfg.window_size / 2
                right = center_x + cfg.window_size / 2
                top = center_y + cfg.window_size / 2
                bottom = center_y - cfg.window_size / 2
                window = from_bounds(left, bottom, right, top, src.transform)
                data = src.read(1, window=window)  # Second band is the FWI mean
                window_transform = src.window_transform(window)

            else:
                raise NotImplementedError(f"Invalid window type: {cfg.window_type}")

        data_resample = fast_resize_with_nan(data, interpolation_y, interpolation_x)
        update_transform = window_transform * window_transform.scale(1 / interpolation_x, 1 / interpolation_y)
        window = from_bounds(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top, update_transform)
        row_start, row_stop = int(window.row_off), int(window.row_off + window.height)
        col_start, col_stop = int(window.col_off), int(window.col_off + window.width)
        data_window = data_resample[row_start:row_stop, col_start:col_stop]

        if data_window.shape[0] != LABEL_RES or data_window.shape[1] != LABEL_RES:
            crop_row = (data_window.shape[0] - LABEL_RES) // 2
            crop_col = (data_window.shape[1] - LABEL_RES) // 2
            data_window = data_window[crop_row : crop_row + LABEL_RES, crop_col : crop_col + LABEL_RES]

        # Check if filling window helps
        median_data = np.nanmedian(data_window)
        data_window = np.nan_to_num(data_window, nan=median_data)

        for i, th in enumerate(cfg.fwi_thresholds):
            fwi_pred = (data_window > th).astype(np.uint8)
            predictions[i].append(fwi_pred.reshape(-1))

        probs = np.stack(
            [
                cfg.fwi_max - np.minimum(data_window.reshape(-1), cfg.fwi_max),
                np.minimum(data_window.reshape(-1), cfg.fwi_max),
            ],
            axis=-1,
        )
        probs = probs / cfg.fwi_max
        tot_probs.append(probs)

    labels = np.concatenate(labels)
    tot_probs = np.concatenate(tot_probs)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for i, th in enumerate(cfg.fwi_thresholds):
        predictions[i] = np.concatenate(predictions[i])

        eval_metrics = get_classification_metrics(predicted=predictions[i], labels=labels, n_classes=2, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics["micro"]
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics["macro"]
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]

        metrics = {
            "macro_Accuracy": macro_acc,
            "macro_Precision": macro_precision,
            "macro_Recall": macro_recall,
            "macro_F1": macro_F1,
            "macro_IOU": macro_IOU,
            "micro_Accuracy": micro_acc,
            "micro_Precision": micro_precision,
            "micro_Recall": micro_recall,
            "micro_F1": micro_F1,
            "micro_IOU": micro_IOU,
            "fire_Accuracy": class_acc[1],
            "fire_Precision": class_precision[1],
            "fire_Recall": class_recall[1],
            "fire_F1": class_F1[1],
            "fire_IOU": class_IOU[1],
        }

        with open(os.path.join(cfg.output_dir, f"{cfg.split}_metrics_{th}.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    micro_auc, macro_auc, class_auc = get_pr_auc_scores(
        scores=tot_probs, labels=labels, n_classes=2, output_dir=Path(cfg.output_dir)
    )

    metrics = {
        "macro_AUC": macro_auc,
        "micro_AUC": micro_auc,
        "fire_AUC": class_auc[1],
    }
    with open(os.path.join(cfg.output_dir, f"{cfg.split}_metrics_auc.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    compute_fwi_baseline()
