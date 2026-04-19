"""File containing visualization of the networks success logic.

Usage Example:
    save_costs(cost_history)
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# folder of this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# go back one step (into the project root), then into the data folder
ROOT_DIR = os.path.dirname(SRC_DIR)

def __smoothen(data: list) -> list:
    """Smoothens data.

    Args:
        data: The original data.

    Returns:
        list: the smoothened data.
    """
    smoothed: list = []
    curr = data[0]

    for x in data:
        curr = 0.05 * x + (1 - 0.05) * curr
        smoothed.append(curr)

    return smoothed


def save_cost(cost_history: list) -> None:
    """Save the cost over epochs as a plot.

    Args:
        cost_history: A list of costs.
    """
    result_dir = os.path.join(ROOT_DIR, "results", "cost_over_epochs.png")

    # create smoothened data
    smooth: list = __smoothen(cost_history)

    # plot
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 5))

    sns.lineplot(data=cost_history, label="Raw", linewidth=1.5, alpha=0.9)
    sns.lineplot(data=smooth, label="Smoothened", linewidth=2.5)

    plt.title("Training Cost Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()

    plt.savefig(result_dir, dpi=200)

def save_accuracy(acc_history: list) -> None:
    """Save the accuracy over epochs as a plot.

    Args:
        acc_history: A list of accuracies.
    """
    result_dir = os.path.join(ROOT_DIR, "results", "accuracy_over_epochs.png")

    # create smoothened data
    smooth: list = __smoothen(acc_history)

    # plot
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 5))

    sns.lineplot(data=acc_history, label="Raw", linewidth=1.5, alpha=0.9)
    sns.lineplot(data=smooth, label="Smoothened", linewidth=2.5)

    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()

    plt.savefig(result_dir, dpi=200)

def save_confusion_matrix(pred: np.ndarray, act: np.ndarray) -> None:
    """Save the confusion matrix using predicted and actual labels.

    Args:
        pred: The predicted labels.
        act: The actual labels.
    """
    result_dir = os.path.join(ROOT_DIR, "results", "confusion_matrix.png")

    # get indices
    pred_indices = np.argmax(pred, axis=1) # shape (batches,)
    true_indices = np.argmax(act, axis=1) # shape (batches,)

    # create empty confusion matrix & then add at the indices
    cm = np.zeros((10, 10), dtype=int)
    for pred, true in zip(pred_indices, true_indices):
        cm[pred][true] += 1

    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 6))

    sns.heatmap(data=cm, annot=True, fmt="d", cmap="Purples")

    plt.title("Confusion Matrix")
    plt.xlabel("Actual Labels")
    plt.ylabel("Predicted Labels")
    plt.tight_layout()

    plt.savefig(result_dir, dpi=200)
