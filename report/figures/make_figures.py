"""Generate the paper's figures (English labels) from results/*.json.

Run from the repository root, after notebooks 01 and 02 (the Grad-CAM figure also needs the model
saved by notebook 01 in models/mobilenet.keras):
    .venv/bin/python report/figures/make_figures.py
Writes PDF figures next to this script.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import DATA_PATH, MODELS_PATH, RESULTS_PATH  # noqa: E402
from src.data import CLASSES, load_dataset, load_lesion_mask      # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})

# Fixed identity per strategy (validated palette); random is the grey dashed reference.
STYLE = {
    "random":      dict(color="#8a8984", linestyle=(0, (4, 2)), label="random"),
    "uncertainty": dict(color="#2a78d6", linestyle="-", label="least confidence"),
    "entropy":     dict(color="#eb6834", linestyle=(0, (1, 1.2)), label="entropy"),
    "margin":      dict(color="#1baf7a", linestyle="-.", label="margin"),
    "malignant":   dict(color="#6250d6", linestyle="-", label="malignant (ours)"),
}


def learning_curves():
    data = json.load(open(RESULTS_PATH / "active_learning.json"))
    results, full = data["results"], data["full_pool"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, metric, title in [(axs[0], "f1_macro", "Macro F1"), (axs[1], "recall_malignant", "Malignant recall")]:
        for strategy, runs in results.items():
            n = np.array(runs[0]["n_labeled"])
            values = np.array([run[metric] for run in runs])
            mean, std = values.mean(axis=0), values.std(axis=0)
            style = STYLE[strategy]
            if strategy in ("random", "margin", "malignant"):   # bands only for a few, to stay readable
                ax.fill_between(n, mean - std, mean + std, color=style["color"], alpha=0.10, linewidth=0)
            ax.plot(n, mean, linewidth=1.4, **style)
        ax.axhline(full[metric], color="#0b0b0b", linestyle=":", linewidth=1, label="whole pool labelled")
        ax.set_xlabel("Number of labelled images")
        ax.set_title(f"{title} on the test set")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlim(20, 270)
    axs[0].set_ylabel("Mean over 20 seeds")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(OUT / "al_learning_curves.pdf")
    plt.close(fig)


def confusion_matrices():
    passive = json.load(open(RESULTS_PATH / "passive_learning.json"))
    y_test = passive["y_test"]
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.85), constrained_layout=True)
    for ax, (name, title) in zip(axs, [("cnn", "CNN (from scratch)"), ("mobilenet", "MobileNetV2")]):
        cm = sum(confusion_matrix(y_test, r["y_pred"], labels=range(3)) for r in passive[name]["runs"])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if share[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(3), CLASSES, fontsize=6)
        ax.set_yticks(range(3), CLASSES if ax is axs[0] else [], fontsize=6)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("True", fontsize=7)
    fig.savefig(OUT / "confusion_passive.pdf")
    plt.close(fig)


def grad_cam_examples():
    """Grad-CAM of the saved MobileNetV2 (seed 1): detected cancers and the two kinds of missed cancers."""
    from src.gpu import setup_gpu
    setup_gpu()
    from src.explain import GradCAM, lesion_overlap, load_model

    _, _, x, _, _, y, names = load_dataset(DATA_PATH, size=128, test_names=True)
    masks = np.array([load_lesion_mask(DATA_PATH, name, 128) for name in names])
    model = load_model(MODELS_PATH / "mobilenet.keras")
    pred = model.predict(x, verbose=0).argmax(axis=1)
    explainer = GradCAM(model)
    malignant = CLASSES.index("malignant")
    detected = np.where((y == malignant) & (pred == malignant))[0]
    missed = np.where((y == malignant) & (pred != malignant))[0]
    energy = {i: lesion_overlap(explainer(x[i])[0], masks[i])["energy"] for i in missed}
    missed = sorted(missed, key=energy.get)

    def distinct(candidates, n):
        """First n images that are not copies of each other (the test set keeps repeated scans together)."""
        chosen = []
        for i in candidates:
            if all(np.corrcoef(x[i].ravel(), x[j].ravel())[0, 1] < 0.8 for j in chosen):
                chosen.append(i)
            if len(chosen) == n:
                break
        return chosen

    rows = [np.random.default_rng(0).choice(detected, 4, replace=False),
            distinct(missed[::-1], 2) + distinct(missed, 2)]   # missed: looked at the lesion / looked elsewhere
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.9), constrained_layout=True)
    for row, indices in zip(axs, rows):
        for ax, i in zip(row, indices):
            cam, probs = explainer(x[i])
            ax.imshow(x[i], cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.contour(masks[i], levels=[0.5], colors="white", linewidths=0.8)
            p = int(np.argmax(probs))
            ax.set_title(("detected" if p == malignant else f"missed: {CLASSES[p]}") + f" ({probs[p]:.0%})", fontsize=7)
            ax.axis("off")
    fig.savefig(OUT / "gradcam_examples.pdf")
    plt.close(fig)


if __name__ == "__main__":
    learning_curves()
    confusion_matrices()
    if (MODELS_PATH / "mobilenet.keras").exists():
        grad_cam_examples()
    print("Figures written to", OUT)
