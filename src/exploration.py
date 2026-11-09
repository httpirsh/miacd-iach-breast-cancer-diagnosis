"""Análise exploratória do dataset (usada em 00_data_exploration.ipynb)."""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data import CLASSES, label_from_name


def list_images(path):
    """Devolve os nomes dos ficheiros e as classes (tiradas do nome, ex.: "benign (12).png")."""
    names = sorted(os.listdir(path))
    return names, [label_from_name(name) for name in names]


def class_counts(labels):
    """Nº de imagens por classe; aceita nomes ou labels numéricas."""
    labels = [CLASSES[l] if not isinstance(l, str) else l for l in labels]
    return {name: labels.count(name) for name in CLASSES}


def plot_class_distribution(counts, title="Distribuição das classes"):
    plt.figure(figsize=(5, 3))
    bars = plt.bar(counts.keys(), counts.values(), color=["seagreen", "steelblue", "firebrick"])
    plt.bar_label(bars)
    plt.ylabel("Nº de imagens")
    plt.title(title)
    plt.show()


def plot_examples(path, names, labels, per_class=4, seed=0):
    """Mostra algumas imagens originais de cada classe."""
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(len(CLASSES), per_class, figsize=(2.5 * per_class, 2.5 * len(CLASSES)))
    for row, class_name in enumerate(CLASSES):
        idx = [i for i, l in enumerate(labels) if l == class_name]
        for col, i in enumerate(rng.choice(idx, size=per_class, replace=False)):
            axs[row, col].imshow(Image.open(os.path.join(path, names[i])).convert("L"), cmap="gray")
            axs[row, col].axis("off")
        axs[row, 0].set_title(class_name, loc="left")
    plt.tight_layout()
    plt.show()


def plot_duplicate_groups(path, groups, max_groups=4):
    """Mostra alguns grupos de imagens repetidas (cada linha é um grupo: a mesma ecografia em ficheiros diferentes)."""
    groups = groups[:max_groups]
    width = max(len(g) for g in groups)
    fig, axs = plt.subplots(len(groups), width, figsize=(3.2 * width, 2.8 * len(groups)), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for row, group in zip(axs, groups):
        for ax, name in zip(row, group):
            ax.imshow(Image.open(os.path.join(path, name)).convert("L"), cmap="gray")
            ax.set_title(name, fontsize=10)
    plt.tight_layout()
    plt.show()
