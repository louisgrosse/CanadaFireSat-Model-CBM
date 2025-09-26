"""Evaluation Script per land cover type for Validation, Test, and Test Hard"""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from DeepSatModels.metrics.numpy_metrics import get_classification_metrics
from src.constants import CONFIG_PATH, LAND_COVER_DICT_IDS, LAND_COVER_FOLDERS
from src.data import get_data
from src.eval.utils import get_pr_auc_scores


def _downsample_lc(lc_data: np.ndarray, out_H: int, out_W: int) -> np.ndarray:
    lc_data = torch.Tensor(lc_data)
    kernel_size = (lc_data.shape[0] // out_H, lc_data.shape[1] // out_W)
    lc_data = F.max_pool2d(lc_data.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size).squeeze(0).squeeze(0)
    lc_data = lc_data.numpy()
    return lc_data


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval")
@torch.no_grad()
def evaluate_lc(cfg: DictConfig):
    """Evaluation Endpoint"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Define output directory
    temp_size = str(cfg["MODEL"]["test_max_seq_len"]) if "test_max_seq_len" in cfg["MODEL"] else "adapt"
    spa_size = str(cfg["MODEL"]["img_res"])
    output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"
    H, W = cfg["MODEL"]["out_H"], cfg["MODEL"]["out_W"]

    # Load outputs
    target_classes = np.load(output_dir / f"{cfg['split']}_target.npy")
    predicted_classes = np.load(output_dir / f"{cfg['split']}_preds.npy")
    probs_classes = np.load(output_dir / f"{cfg['split']}_probs.npy")

    # Get Dataset
    datamodule = get_data(cfg)
    dataset = datamodule.test_dataloader(split=cfg["split"]).dataset

    tot_lc_data = []
    for i in tqdm(range(len(dataset)), desc=f"Extracting Land Cover on split {cfg['split']}", total=len(dataset)):

        data = dataset[i]

        if cfg["mode"] == "image" or cfg["MODEL"]["architecture"] in ["EnvResNet", "EnvViTFactorizeModel"]:
            img_name_info = data[1]
        else:
            img_name_info = data[2]

        tile_dir = img_name_info["tile_id"]
        lc_dir = LAND_COVER_FOLDERS[0 if img_name_info["file_id"] == "POS" else 1]

        lc_data = np.load(Path(lc_dir) / tile_dir / "lc.npy")
        lc_data = _downsample_lc(lc_data, H, W)
        tot_lc_data.append(lc_data.reshape(-1))

    tot_lc_data = np.concatenate(tot_lc_data, axis=0)
    assert tot_lc_data.shape[0] == target_classes.shape[0], "Land Cover data and target classes do not match"

    lc_metrics = {}

    for lc in np.sort(np.unique(tot_lc_data)):

        idx = np.where(tot_lc_data == lc)

        lc_preds = predicted_classes[idx]
        lc_labels = target_classes[idx]
        lc_probs = probs_classes[idx]
        print(lc_preds.shape, lc_labels.shape, lc_probs.shape)
        sample_size = lc_preds.shape[0] / (H * W)
        tot_patch = lc_preds.shape[0]
        pos_patch = lc_labels.sum()
        f1_random = 2 * (pos_patch / tot_patch) / (1 + pos_patch / tot_patch)

        # Compute metrics
        eval_metrics = get_classification_metrics(
            predicted=lc_preds,
            labels=lc_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
            unk_masks=None,
        )

        _, _, class_auc = get_pr_auc_scores(
            scores=lc_probs,
            labels=lc_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
            output_dir=output_dir,
        )

        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]
        weighted_f1_full = (float(class_F1[1]) - f1_random) / (1 - f1_random)

        metrics = {
            "fire_Accuracy": float(class_acc[1]),
            "fire_Precision": float(class_precision[1]),
            "fire_Recall": float(class_recall[1]),
            "fire_F1": float(class_F1[1]),
            "fire_IOU": float(class_IOU[1]),
            "fire_AUC": float(class_auc[1]),
            "sample_size": sample_size,
            "f1_random": f1_random,
            "tot_fire_F1_weighted": weighted_f1_full,
        }

        lc_metrics[LAND_COVER_DICT_IDS[lc]] = metrics

    with open(output_dir / f"{cfg['split']}_lc_metrics.json", "w") as f:
        json.dump(lc_metrics, f, indent=4)


if __name__ == "__main__":
    evaluate_lc()
