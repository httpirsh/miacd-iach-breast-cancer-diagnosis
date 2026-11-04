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


def plot_learning_curves(results, metric, title, reference=None, ylabel=None):
    """Métrica no teste em função do nº de imagens anotadas: média das seeds (linha) ± desvio padrão (sombra).

    results: {estratégia: [history de cada seed]}; reference: valor obtido com todas as imagens anotadas.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, runs in results.items():
        n_labeled = runs[0]["n_labeled"]
        values = np.array([run[metric] for run in runs])
        mean, std = values.mean(axis=0), values.std(axis=0)
        style = {"color": "gray", "linestyle": "dashed"} if strategy == "random" else {}
        ax.plot(n_labeled, mean, label=strategy, **style)
        ax.fill_between(n_labeled, mean - std, mean + std, alpha=0.15, color=style.get("color"))
    if reference is not None:
        ax.axhline(reference, color="black", linestyle="dotted", label="todas as imagens anotadas")
    ax.set_xlabel("Nº de imagens anotadas")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def plot_predictions(images, y_true, y_pred, indices, title, per_row=6):
    """Mostra imagens do teste com a classe real e a prevista (título verde = certo, vermelho = errado)."""
    indices = list(indices)
    if not indices:
        print(f"{title}: nenhuma imagem")
        return
    rows = (len(indices) + per_row - 1) // per_row
    fig, axs = plt.subplots(rows, per_row, figsize=(2.6 * per_row, 3.3 * rows), squeeze=False)
    fig.suptitle(title)
    for ax in axs.flat:
        ax.axis("off")
    for ax, i in zip(axs.flat, indices):
        ax.imshow(images[i], cmap="gray")
        correct = y_true[i] == y_pred[i]
        ax.set_title(f"real: {CLASSES[y_true[i]]}\nprevisto: {CLASSES[y_pred[i]]}", fontsize=9,
                     color="green" if correct else "red")
    plt.tight_layout()
    plt.show()
