"""Evaluation Script across multiple probability threshold for Validation, Test, and Test Hard (needs to run eval.py)"""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from DeepSatModels.metrics.numpy_metrics import get_classification_metrics
from src.constants import CONFIG_PATH


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval")
@torch.no_grad()
def evaluate_th(cfg: DictConfig):
    """Evaluation Endpoint"""

    # Define output directory
    temp_size = str(cfg["MODEL"]["test_max_seq_len"]) if "test_max_seq_len" in cfg["MODEL"] else "adapt"
    spa_size = str(cfg["MODEL"]["img_res"]) if "img_res" in cfg["MODEL"] else str(cfg["MODEL"]["mid_input_res"])

    if "hard" in cfg["DATASETS"]["eval"]["paths"]:
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}_hard"
    else:
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"

    H, W = cfg["MODEL"]["out_H"], cfg["MODEL"]["out_W"]

    # Load outputs
    target_classes = np.load(output_dir / f"{cfg['split']}_target.npy")
    probs_classes = np.load(output_dir / f"{cfg['split']}_probs.npy")

    f1_results = []
    pos_f1_results = []
    tot_results = {}
    tot_pos_results = {}
    thresholds = np.linspace(0.1, 0.9, 9)

    for th in tqdm(thresholds):

        preds = (probs_classes[:, 1] > th).astype(int)

        # Positive & Negative bucket evaliation
        th_metrics = get_classification_metrics(
            predicted=preds,
            labels=target_classes,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
            unk_masks=None,
        )

        th_micro_acc, th_micro_precision, th_micro_recall, th_micro_F1, th_micro_IOU = th_metrics["micro"]
        th_macro_acc, th_macro_precision, th_macro_recall, th_macro_F1, th_macro_IOU = th_metrics["macro"]
        th_class_acc, th_class_precision, th_class_recall, th_class_F1, th_class_IOU = th_metrics["class"]

        th_metrics = {
            "macro_Accuracy": float(th_macro_acc),
            "macro_Precision": float(th_macro_precision),
            "macro_Recall": float(th_macro_recall),
            "macro_F1": float(th_macro_F1),
            "macro_IOU": float(th_macro_IOU),
            "micro_Accuracy": float(th_micro_acc),
            "micro_Precision": float(th_micro_precision),
            "micro_Recall": float(th_micro_recall),
            "micro_F1": float(th_micro_F1),
            "micro_IOU": float(th_micro_IOU),
            "fire_Accuracy": float(th_class_acc[1]),
            "fire_Precision": float(th_class_precision[1]),
            "fire_Recall": float(th_class_recall[1]),
            "fire_F1": float(th_class_F1[1]),
            "fire_IOU": float(th_class_IOU[1]),
        }

        tot_results[f"th_{th}"] = th_metrics
        f1_results.append(th_metrics["fire_F1"])

        # Positive only evaluation
        pos_labels = target_classes.reshape(-1, H, W)
        pos_preds = preds.reshape(-1, H, W)

        pos_idx = np.where(pos_labels.any(axis=(1, 2)))
        pos_preds = pos_preds[pos_idx].reshape(-1)
        pos_labels = pos_labels[pos_idx].reshape(-1)

        pos_metrics = get_classification_metrics(
            predicted=pos_preds,
            labels=pos_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
            unk_masks=None,
        )

        pos_micro_acc, pos_micro_precision, pos_micro_recall, pos_micro_F1, pos_micro_IOU = pos_metrics["micro"]
        pos_macro_acc, pos_macro_precision, pos_macro_recall, pos_macro_F1, pos_macro_IOU = pos_metrics["macro"]
        pos_class_acc, pos_class_precision, pos_class_recall, pos_class_F1, pos_class_IOU = pos_metrics["class"]

        pos_metrics = {
            "macro_Accuracy": float(pos_macro_acc),
            "macro_Precision": float(pos_macro_precision),
            "macro_Recall": float(pos_macro_recall),
            "macro_F1": float(pos_macro_F1),
            "macro_IOU": float(pos_macro_IOU),
            "micro_Accuracy": float(pos_micro_acc),
            "micro_Precision": float(pos_micro_precision),
            "micro_Recall": float(pos_micro_recall),
            "micro_F1": float(pos_micro_F1),
            "micro_IOU": float(pos_micro_IOU),
            "fire_Accuracy": float(pos_class_acc[1]),
            "fire_Precision": float(pos_class_precision[1]),
            "fire_Recall": float(pos_class_recall[1]),
            "fire_F1": float(pos_class_F1[1]),
            "fire_IOU": float(pos_class_IOU[1]),
        }
        tot_pos_results[f"th_{th}"] = pos_metrics
        pos_f1_results.append(pos_metrics["fire_F1"])

    fig, ax = plt.subplots()
    ax.plot(thresholds, f1_results)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold")
    fig.savefig(str(output_dir / f"{cfg['split']}_f1_vs_threshold.png"))

    fig, ax = plt.subplots()
    ax.plot(thresholds, pos_f1_results)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("Positive F1 Score vs Threshold")
    fig.savefig(str(output_dir / f"{cfg['split']}_pos_f1_vs_threshold.png"))

    with open(output_dir / f"{cfg['split']}_threshold_metrics.json", "w") as f:
        json.dump(tot_results, f, indent=4)

    with open(output_dir / f"{cfg['split']}_pos_threshold_metrics.json", "w") as f:
        json.dump(tot_pos_results, f, indent=4)


if __name__ == "__main__":
    evaluate_th()
