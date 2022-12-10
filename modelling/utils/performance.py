from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report


def plot_precision_recall_curve_over_thresholds(
    precisions: np.array,
    recalls: np.array,
    thresholds: np.array,
    figsize: Tuple[int, int] = (16, 8),
    fontsize: int = 14,
    fontsize_title: int = 16,
    grid: bool = True,
):

    _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(thresholds, precisions[:-1], linestyle="-", color="blue", label="precision")
    ax.plot(thresholds, recalls[:-1], linestyle="--", color="green", label="recall")

    ax.set_xlabel("thresholds", fontsize=fontsize)

    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_title("Precision recall curves over thresholds", fontsize=fontsize_title)

    ax.grid(grid)
    ax.legend()

    ax.set_yticks(np.arange(0, 1.1, 0.1))

    return ax


def plot_precision_recall_curve(
    precisions: np.array,
    recalls: np.array,
    figsize: Tuple[int, int] = (16, 8),
    fontsize: int = 14,
    fontsize_title: int = 16,
    grid: bool = True,
):

    _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(recalls, precisions)

    ax.set_xlabel("recall", fontsize=fontsize)
    ax.set_ylabel("precision", fontsize=fontsize)

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.set_title("Precision recall curve", fontsize=fontsize_title)

    ax.grid(grid)

    return ax


def plot_conf_matrix(
    conf_matrix: np.array,
    figsize: Tuple = (6, 4),
    classes: Optional[List] = None,
    **kwargs
):
    _, ax = plt.subplots(1, 1, figsize=figsize)
    if classes:
        kwargs["xticklabels"] = classes
        kwargs["yticklabels"] = classes
    sns.heatmap(conf_matrix, annot=True, fmt=".0f", ax=ax, **kwargs)

    return ax


def get_classification_report(
    y_true, y_pred, digits: int = 4, classes: Optional[List] = None, **kwargs
):
    if classes:
        kwargs["target_names"] = classes
    print(classification_report(y_true=y_true, y_pred=y_pred, digits=digits, **kwargs))
