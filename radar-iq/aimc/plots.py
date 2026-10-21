from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_examples(images, labels, classes, path):
    cols = min(4, len(classes))
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows), squeeze=False)
    for label, axis in enumerate(axes.flat):
        if label >= len(classes):
            axis.axis("off")
            continue
        i = np.flatnonzero(labels == label)[0]
        axis.imshow(images[i, 0].numpy(), origin="lower", aspect="auto", cmap="viridis",
                    extent=[0, 1, -0.5, 0.5], vmin=0, vmax=1)
        axis.set(title=classes[label], xlabel="Pulse time (normalized)", ylabel="Frequency / fs")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [row["epoch"] for row in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot(epochs, [row[f"{split}_accuracy"] for row in history], label=split)
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(matrix, classes, path):
    size = max(8, len(classes) * 0.45)
    fig, axis = plt.subplots(figsize=(size, max(7, size)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)),
             xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="True")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      fontsize=7 if len(classes) > 16 else 10,
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
