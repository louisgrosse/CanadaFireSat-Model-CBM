"""Utils functions for the evaluation"""
import os
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

def get_classification_metrics_sklearn(
    predicted: np.ndarray, labels: np.ndarray, n_classes: int, unk_masks: Optional[np.ndarray] = None
):
    """Compute set of evaluation metrics"""

    if unk_masks is not None:
        predicted = predicted[unk_masks]
        labels = labels[unk_masks]

    # Compute class metrics and macro metrics
    output_dict = classification_report(labels, predicted, output_dict=True, zero_division=0)
    macro_acc = sum([output_dict[str(i)]["recall"] for i in range(n_classes)]) / n_classes
    macro_iou = jaccard_score(labels, predicted, average="macro")

    # Compute micro metrics
    micro_precision = precision_score(labels, predicted, average="micro")
    micro_recall = recall_score(labels, predicted, average="micro")
    micro_f1 = f1_score(labels, predicted, average="micro")
    micro_iou = jaccard_score(labels, predicted, average="micro")

    # Compute class IoU metrics
    labels = np.eye(n_classes)[labels]
    predicted = np.eye(n_classes)[predicted]
    iou_scores = []
    for i in range(n_classes):
        iou_scores.append(jaccard_score(labels[:, i], predicted[:, i]), average="binary")

    return {
        "class": [
            [output_dict[str(i)]["recall"] for i in range(n_classes)],
            [output_dict[str(i)]["precision"] for i in range(n_classes)],
            [output_dict[str(i)]["recall"] for i in range(n_classes)],
            [output_dict[str(i)]["f1-score"] for i in range(n_classes)],
            iou_scores,
        ],
        "micro": [output_dict["accuracy"], micro_precision, micro_recall, micro_f1, micro_iou],
        "macro": [
            macro_acc,
            output_dict["macro avg"]["precision"],
            output_dict["macro avg"]["recall"],
            output_dict["macro avg"]["f1-score"],
            macro_iou,
        ],
    }


def get_pr_auc_scores(
    scores: np.ndarray, labels: np.ndarray, n_classes: int, output_dir: Optional[os.PathLike] = None
) -> Tuple[float, float, List[float]]:
    """Compute PRAUC and Plot"""

    labels = labels.astype(int)

    if len(labels.shape) == 1 or labels.shape[1] == 1:
        labels = np.eye(n_classes)[labels]

    if scores.shape[1] == 1 and n_classes == 2:
        scores = np.concatenate([1 - scores, scores], axis=1)

    elif scores.shape[1] != n_classes:
        raise ValueError(f"Scores shape {scores.shape} does not match n_classes {n_classes}")

    macro_pr_auc_score = average_precision_score(labels, scores, average="macro")
    micro_pr_auc_score = average_precision_score(labels, scores, average="micro")

    pr_auc_scores = []
    for i in range(n_classes):
        pr_auc_scores.append(average_precision_score(labels[:, i], scores[:, i]))

    if output_dir is not None:
        # Plot PR curve for last class (positive class)
        precision, recall, _ = precision_recall_curve(labels[:, -1], scores[:, -1])
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precison-Recall Curve: AUC = {pr_auc_scores[-1]:.2f}")
        plt.legend()
        plt.grid()
        plt.savefig(output_dir / "precision_recall_curve.png")

    return micro_pr_auc_score, macro_pr_auc_score, pr_auc_scores
