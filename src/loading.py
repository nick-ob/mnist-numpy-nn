"""File containing logic of loading raw MNIST data into usable numpy arrays.

Usage Example:

    (x_train, y_train), (x_test, y_test) = load_mnist()
"""
import os
import gzip as gz
import numpy as np

def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load the MNIST dataset from raw .gz files.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray]]: Contains training and testing data,
        which each contain images and labels.
    """

    def load_images(path: str) -> np.ndarray:
        """Helper function. Loads the images and flattens them.

        Args:
            path: Path to image data.

        Returns:
            np.ndarray: Flattened image data.
        """
        with gz.open(path, "rb") as file:
            # first 16 bytes header
            file.read(16)

            data: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8)

        # flatten data and normalize (input is originally 0-255) to values from 0 to 1
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    def load_labels(path: str) -> np.ndarray:
        """Helper function. Loads the labels and one-hot encodes them.

        Args:
            path: Path to label data.

        Returns:
            np.ndarray: One-hot encoded image data.
        """
        with gz.open(path, "rb") as file:
            # first 8 bytes header
            file.read(8)

            data: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8)

        # one-hot encode using an identity matrix
        return np.eye(10)[data]

    # folder of this file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # go back one step (into the project root), then into the data folder
    root_dir = os.path.dirname(src_dir)
    mnist_dir = os.path.join(root_dir, "data", "mnist")
    os.makedirs(mnist_dir, exist_ok=True)

    # make sure all raw files exist
    required_files = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    }

    existing_files = set(os.listdir(mnist_dir))

    # take difference of sets (all elements in required but not in existing)
    missing = required_files - existing_files

    if missing:
        raise FileNotFoundError(
            f"MNIST data does not exist in {mnist_dir}. Missing: {missing}\n"
            f"Try running data/loader.py (python data/loader.py) to install the MNIST data."
        )

    # load data into np.arrays
    x_train: np.ndarray = load_images(os.path.join(mnist_dir, "train-images-idx3-ubyte.gz"))
    y_train: np.ndarray = load_labels(os.path.join(mnist_dir, "train-labels-idx1-ubyte.gz"))

    x_test: np.ndarray = load_images(os.path.join(mnist_dir, "t10k-images-idx3-ubyte.gz"))
    y_test: np.ndarray = load_labels(os.path.join(mnist_dir, "t10k-labels-idx1-ubyte.gz"))

    return ((x_train, y_train), (x_test, y_test))
