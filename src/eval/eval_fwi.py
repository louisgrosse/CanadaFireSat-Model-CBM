"""Evaluation Script per FWI-Bucket for Validation, Test, and Test Hard (needs to run eval.py)"""
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from DeepSatModels.metrics.numpy_metrics import get_classification_metrics
from src.constants import CONFIG_PATH
from src.eval.utils import get_pr_auc_scores


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval")
@torch.no_grad()
def evaluate_fwi(cfg: DictConfig):
    """Evaluation Endpoint"""

    # Define output directory
    temp_size = str(cfg["MODEL"]["test_max_seq_len"]) if "test_max_seq_len" in cfg["MODEL"] else "adapt"
    spa_size = str(cfg["MODEL"]["img_res"]) if "img_res" in cfg["MODEL"] else str(cfg["MODEL"]["mid_input_res"])
    output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"
    H, W = cfg["MODEL"]["out_H"], cfg["MODEL"]["out_W"]

    # Load outputs
    target_classes = np.load(output_dir / f"{cfg['split']}_target.npy")
    predicted_classes = np.load(output_dir / f"{cfg['split']}_preds.npy")
    probs_classes = np.load(output_dir / f"{cfg['split']}_probs.npy")
    fwis = np.load(output_dir / f"{cfg['split']}_fwi.npy")

    # Positive only evaluation
    pos_labels = target_classes.reshape(-1, H, W)
    pos_preds = predicted_classes.reshape(-1, H, W)
    pos_probs = probs_classes.reshape(
        -1, H, W, cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
    )

    pos_idx = np.where(pos_labels.any(axis=(1, 2)))
    pos_preds = pos_preds[pos_idx].reshape(-1)
    pos_labels = pos_labels[pos_idx].reshape(-1)
    pos_probs = pos_probs[pos_idx].reshape(
        -1, cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
    )

    pos_metrics = get_classification_metrics(
        predicted=pos_preds,
        labels=pos_labels,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        unk_masks=None,
    )
    pos_micro_auc, pos_macro_auc, pos_class_auc = get_pr_auc_scores(
        scores=pos_probs,
        labels=pos_labels,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
    )

    pos_micro_acc, pos_micro_precision, pos_micro_recall, pos_micro_F1, pos_micro_IOU = pos_metrics["micro"]
    pos_macro_acc, pos_macro_precision, pos_macro_recall, pos_macro_F1, pos_macro_IOU = pos_metrics["macro"]
    pos_class_acc, pos_class_precision, pos_class_recall, pos_class_F1, pos_class_IOU = pos_metrics["class"]

    pos_metrics = {
        "macro_Accuracy": pos_macro_acc,
        "macro_Precision": pos_macro_precision,
        "macro_Recall": pos_macro_recall,
        "macro_F1": pos_macro_F1,
        "macro_IOU": pos_macro_IOU,
        "macro_AUC": pos_macro_auc,
        "micro_Accuracy": pos_micro_acc,
        "micro_Precision": pos_micro_precision,
        "micro_Recall": pos_micro_recall,
        "micro_F1": pos_micro_F1,
        "micro_IOU": pos_micro_IOU,
        "micro_AUC": pos_micro_auc,
        "fire_Accuracy": pos_class_acc[1],
        "fire_Precision": pos_class_precision[1],
        "fire_Recall": pos_class_recall[1],
        "fire_F1": pos_class_F1[1],
        "fire_IOU": pos_class_IOU[1],
        "fire_AUC": pos_class_auc[1],
    }

    with open(output_dir / f"{cfg['split']}_pos_only_metrics.txt", "w") as f:
        for key, value in pos_metrics.items():
            f.write(f"{key}: {value}\n")

    # Collecting FWI Negative
    neg_results = {}
    cfg["fwi_ths"].append("inf")
    for k, fwi_th in enumerate(cfg["fwi_ths"]):
        b_fwi_th = cfg["fwi_ths"][k - 1] if k > 0 else 0

        if fwi_th == "inf":
            idx = np.where(fwis >= b_fwi_th)
        else:
            idx = np.where((fwis >= b_fwi_th) & (fwis < fwi_th))

        fwi_preds = predicted_classes[idx]
        fwi_labels = target_classes[idx]
        fwi_probs = probs_classes[idx]
        sample_size = fwi_preds.shape[0] / (H * W)
        tot_patch = fwi_preds.shape[0]

        # Positive & Negative bucket evaliation
        fwi_metrics = get_classification_metrics(
            predicted=fwi_preds,
            labels=fwi_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
            unk_masks=None,
        )
        fwi_micro_auc, fwi_macro_auc, fwi_class_auc = get_pr_auc_scores(
            scores=fwi_probs,
            labels=fwi_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
        )

        fwi_micro_acc, fwi_micro_precision, fwi_micro_recall, fwi_micro_F1, fwi_micro_IOU = fwi_metrics["micro"]
        fwi_macro_acc, fwi_macro_precision, fwi_macro_recall, fwi_macro_F1, fwi_macro_IOU = fwi_metrics["macro"]
        fwi_class_acc, fwi_class_precision, fwi_class_recall, fwi_class_F1, fwi_class_IOU = fwi_metrics["class"]

        fwi_metrics = {
            "macro_Accuracy": fwi_macro_acc,
            "macro_Precision": fwi_macro_precision,
            "macro_Recall": fwi_macro_recall,
            "macro_F1": fwi_macro_F1,
            "macro_IOU": fwi_macro_IOU,
            "macro_AUC": fwi_macro_auc,
            "micro_Accuracy": fwi_micro_acc,
            "micro_Precision": fwi_micro_precision,
            "micro_Recall": fwi_micro_recall,
            "micro_F1": fwi_micro_F1,
            "micro_IOU": fwi_micro_IOU,
            "micro_AUC": fwi_micro_auc,
            "fire_Accuracy": fwi_class_acc[1],
            "fire_Precision": fwi_class_precision[1],
            "fire_Recall": fwi_class_recall[1],
            "fire_F1": fwi_class_F1[1],
            "fire_IOU": fwi_class_IOU[1],
            "fire_AUC": fwi_class_auc[1],
            "sample_size": sample_size,
        }

        with open(output_dir / f"{cfg['split']}_fwi_{fwi_th}_metrics.txt", "w") as f:
            for key, value in fwi_metrics.items():
                f.write(f"{key}: {value}\n")

        # Positive only evaluation
        fwi_labels = fwi_labels.reshape(-1, H, W)
        fwi_preds = fwi_preds.reshape(-1, H, W)
        fwi_probs = fwi_probs.reshape(
            -1,
            H,
            W,
            cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        )

        pos_idx = np.where(fwi_labels.any(axis=(1, 2)))
        pos_fwi_preds = fwi_preds[pos_idx].reshape(-1)
        pos_fwi_labels = fwi_labels[pos_idx].reshape(-1)
        pos_fwi_probs = fwi_probs[pos_idx].reshape(
            -1, cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
        )
        sample_size = pos_fwi_preds.shape[0] / (H * W)
        pos_patch = pos_fwi_labels.sum()
        f1_random = 2 * (pos_patch / tot_patch) / (1 + pos_patch / tot_patch)
        weighted_f1_full = (fwi_class_F1[1] - f1_random) / (1 - f1_random)

        fwi_metrics = get_classification_metrics(
            predicted=pos_fwi_preds,
            labels=pos_fwi_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
            unk_masks=None,
        )
        fwi_micro_auc, fwi_macro_auc, fwi_class_auc = get_pr_auc_scores(
            scores=pos_fwi_probs,
            labels=pos_fwi_labels,
            n_classes=(
                cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
            ),
        )

        fwi_micro_acc, fwi_micro_precision, fwi_micro_recall, fwi_micro_F1, fwi_micro_IOU = fwi_metrics["micro"]
        fwi_macro_acc, fwi_macro_precision, fwi_macro_recall, fwi_macro_F1, fwi_macro_IOU = fwi_metrics["macro"]
        fwi_class_acc, fwi_class_precision, fwi_class_recall, fwi_class_F1, fwi_class_IOU = fwi_metrics["class"]

        fwi_metrics = {
            "macro_Accuracy": fwi_macro_acc,
            "macro_Precision": fwi_macro_precision,
            "macro_Recall": fwi_macro_recall,
            "macro_F1": fwi_macro_F1,
            "macro_IOU": fwi_macro_IOU,
            "macro_AUC": fwi_macro_auc,
            "micro_Accuracy": fwi_micro_acc,
            "micro_Precision": fwi_micro_precision,
            "micro_Recall": fwi_micro_recall,
            "micro_F1": fwi_micro_F1,
            "micro_IOU": fwi_micro_IOU,
            "micro_AUC": fwi_micro_auc,
            "fire_Accuracy": fwi_class_acc[1],
            "fire_Precision": fwi_class_precision[1],
            "fire_Recall": fwi_class_recall[1],
            "fire_F1": fwi_class_F1[1],
            "fire_IOU": fwi_class_IOU[1],
            "fire_AUC": fwi_class_auc[1],
            "sample_size": sample_size,
            "f1_random": f1_random,
            "tot_fire_F1_weighted": weighted_f1_full,
        }

        with open(output_dir / f"pos_{cfg['split']}_fwi_{fwi_th}_metrics.txt", "w") as f:
            for key, value in fwi_metrics.items():
                f.write(f"{key}: {value}\n")

        # Negative only evaluation
        neg_idx = np.where(~fwi_labels.any(axis=(1, 2)))
        neg_fwi_preds = fwi_preds[neg_idx].reshape(-1)
        sample_size = neg_fwi_preds.shape[0] / (H * W)
        neg_results[f"fwi_{fwi_th}"] = {
            "sample_size": sample_size,
            "pred_ratio": neg_fwi_preds.sum() / neg_fwi_preds.shape[0],
        }

    with open(output_dir / f"neg_{cfg['split']}_fwi_metrics.txt", "w") as f:
        for key, value in neg_results.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    evaluate_fwi()
