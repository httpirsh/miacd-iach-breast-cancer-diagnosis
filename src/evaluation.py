"""Métricas e gráficos comuns às experiências."""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .data import CLASSES

MALIGNANT = CLASSES.index("malignant")


def compute_metrics(y_true, y_pred):
    """Accuracy, recall/F1 macro, recall de maligno e falsos negativos (maligno previsto como outra classe)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_malignant": float(recall_score(y_true == MALIGNANT, y_pred == MALIGNANT, zero_division=0)),
        "false_negatives": int(np.sum((y_true == MALIGNANT) & (y_pred != MALIGNANT))),
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    confusion_mtx = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    f, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="BuPu", linecolor="gray", fmt="d", ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def plot_history(results, title, baseline=None):
    """Accuracy e loss por iteração de Active Learning, para cada batch size.

    results: {batch_size: history}; baseline: {"accuracy_train": x, "accuracy_test": y} do Passive Learning.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    colors = ["darkred", "dodgerblue", "green", "orange"]

    for ax, split, name in [(axs[0], "train", "Train set"), (axs[1], "test", "Test set")]:
        for color, (batch_size, history) in zip(colors, results.items()):
            iterations = range(len(history[f"accuracy_{split}"]))
            ax.plot(iterations, history[f"accuracy_{split}"], label=f"accuracy - batch of {batch_size}", color=color)
            ax.plot(iterations, history[f"loss_{split}"], label=f"loss - batch of {batch_size}", color=color,
                    linestyle="dashed", linewidth=0.7)
        if baseline is not None:
            ax.axhline(baseline[f"accuracy_{split}"], color="black", linestyle="dashed", linewidth=1,
                       label="passive learning")
        ax.set_xlabel("Active Learning Iterations")
        ax.set_ylabel("Accuracy/Loss")
        ax.set_title(name)
        ax.set_ylim(0, 1.5)

    axs[1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.show()
