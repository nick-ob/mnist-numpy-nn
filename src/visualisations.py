"""File containing visualization of the networks success logic.

Usage Example:
    save_loss(loss_history)
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt

def save_loss(cost_history: list) -> None:
    """Save the cost over epochs as a plot.

    Args:
        loss_history: A list of losses.
    """
    def smoothen(data: list) -> list:
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

    # folder of this file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # go back one step (into the project root), then into the data folder
    root_dir = os.path.dirname(src_dir)
    result_dir = os.path.join(root_dir, "results", "cost_over_epochs.png")

    # create smoothened data
    smooth: list = smoothen(cost_history)

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
