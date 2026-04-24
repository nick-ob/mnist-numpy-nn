"""File containing visualization of the networks success logic.

Usage Example:
    save_costs(cost_history, "my save")
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# folder of this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# go back one step (into the project root), then into the data folder
ROOT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def save_cost(cost_history: list, name: str) -> None:
    """Save the cost over iterations as a plot.

    Args:
        cost_history: A list of costs.
        name: The name of the folder the plot should be saved under.
    """
    save_dir = os.path.join(RESULTS_DIR, name)
    file_dir = os.path.join(save_dir, "cost_over_iterations.png")
    os.makedirs(save_dir, exist_ok=True)
    # create smoothened data
    smooth: list = __smoothen(cost_history)

    # plot
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 5))

    sns.lineplot(data=cost_history, label="Raw", linewidth=1, alpha=0.7, color="purple")
    sns.lineplot(data=smooth, label="Smoothened", linewidth=1.5, color="black")

    plt.title("Training Cost Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    plt.xscale("log")

    plt.savefig(file_dir, dpi=200)
    plt.close()

def save_accuracy(acc_history: list, acc_test: float, name: str) -> None:
    """Save the accuracy over iterations as a plot.

    Args:
        acc_history: A list of accuracies.
        acc_test: The final testing accuracy.
        name: The name of the folder the plot should be saved under.
    """
    save_dir = os.path.join(RESULTS_DIR, name)
    file_dir = os.path.join(save_dir, "accuracy_over_iterations.png")
    os.makedirs(save_dir, exist_ok=True)
    # create smoothened data
    smooth: list = __smoothen(acc_history)

    # plot
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 5))

    sns.lineplot(data=acc_history, label="Raw", linewidth=1, alpha=0.7, color="purple")
    sns.lineplot(data=smooth, label="Smoothened", linewidth=1.5, color="black")
    plt.axhline(
            y=acc_history[-1], label=f"Final training accuracy ({acc_history[-1]}%)",
            color="lightblue", linestyle="--", linewidth=1.5
    )
    plt.axhline(
            y=acc_test, label=f"Testing accuracy ({acc_test})%",
            color="pink", linestyle="--", linewidth=1.5
    )

    plt.title("Training Accuracy Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.xscale("log")

    plt.savefig(file_dir, dpi=200)
    plt.close()

def save_confusion_matrix(pred: np.ndarray, act: np.ndarray, name: str) -> None:
    """Save the confusion matrix using predicted and actual labels.

    Args:
        pred: The predicted labels.
        act: The actual labels.
        name: The name of the folder the plot should be saved under.
    """
    save_dir = os.path.join(RESULTS_DIR, name)
    file_dir = os.path.join(save_dir, "confusion_matrix.png")
    os.makedirs(save_dir, exist_ok=True)

    # get indices
    pred_indices = np.argmax(pred, axis=1) # shape (batches,)
    true_indices = np.argmax(act, axis=1) # shape (batches,)

    # create empty confusion matrix & then add at the indices
    cm = np.zeros((10, 10), dtype=int)
    for predicted, true in zip(pred_indices, true_indices):
        cm[predicted][true] += 1

    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(9, 6))

    sns.heatmap(data=cm, annot=True, fmt="d", cmap="Purples")

    plt.title("Confusion Matrix")
    plt.xlabel("Actual Labels")
    plt.ylabel("Predicted Labels")
    plt.tight_layout()

    plt.savefig(file_dir, dpi=200)
    plt.close()

def save_classified(x: np.ndarray, pred: np.ndarray, act: np.ndarray, name: str) -> None:
    """Save a collection of correctly classified and misclassified images
    .Showing the image and their predicted + actual labels.

    Args:
        x: The input data that the labels were predicted with.
        pred: The predicted labels.
        act: The actual labels.
        name: The name of the folder the plot should be saved under.
    """
    pred_indices = np.argmax(pred, axis=1)
    true_indices = np.argmax(act, axis=1)

    save_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    def plot_samples(
            x: np.ndarray,
            indices: np.ndarray, pred_indices: np.ndarray, true_indices: np.ndarray,
            title: str, file_path: str
    ) -> None:
        """Plot a set of classification examples.

        Args:
            x: The input data that the labels were predicted with.
            indices: The indices of the images.
            pred_indices: The indices of the predicted labels.
            true_indices: The indices of the correct labels.
            title: The plot title to use.
            file_path: The path to save to.
        """
        selected_indices = np.random.choice(
            indices,
            size=min(8, len(indices)),
            replace=False
        )

        plt.figure(figsize=(12, 6))

        for i, idx in enumerate(selected_indices):
            plt.subplot(2, 4, i + 1)

            image = x[idx].reshape(28, 28)
            plt.imshow(image, cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.title(f"Pred: {pred_indices[idx]} | True: {true_indices[idx]}")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(file_path, dpi=200)
        plt.close()

    misclassified_indices = np.where(pred_indices != true_indices)[0]
    if len(misclassified_indices) != 0:
        file_dir = os.path.join(save_dir, "misclassifications.png")
        plot_samples(x,
                     misclassified_indices, pred_indices, true_indices,
                     "Misclassified Examples", file_dir
        )

    classified_indices = np.where(pred_indices == true_indices)[0]
    if len(classified_indices) != 0:
        file_dir = os.path.join(save_dir, "correctly_classified.png")
        plot_samples(x,
                     classified_indices, pred_indices, true_indices,
                     "Correctly Classified Examples", file_dir
        )
