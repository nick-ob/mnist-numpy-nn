"""File containing the logic of the network, represented as a class.

Usage example:

    network = Network(3, 2, 2)
    x_train = np.array([1, 0, 1])
    y_train = np.array([[1]])
    network.train((x_train, y_train), 0.01, 100)
"""
import os
import numpy as np
from src.loss import CCE, accuracy
from src.layer import Layer
from src.activations import ReLu, Softmax

class Network:
    """A class representing the actual network.

    Pieces together the individual parts (layers, activations, loss) to get a running network.

    Attributes:
        __arch: The architecture of the network, meaning the layers and their node amounts.
        __layers: The layers of the network, including activation layers.
        __history: A variable to cache the training history. Needed to save in the save function.
    """
    def __init__(self, *nodes: int) -> None:
        """Initialises instances using node amounts.

        Args:
            nodes: All node counts. Each node count represents the amount of nodes of one layer.
        """
        self.__arch: tuple[int] = nodes
        self.__layers = self.__init_layers(nodes)
        self.__history: dict[str, list] = {}

    def __init_layers(self, nodes: tuple[int]) -> list:
        """Takes a tuple containing layer nodes and creates a list of actual layers.

        Args:
            nodes: Tuple of integers representing the layers and their node counts.

        Returns:
            list: List of layers representing the network.
        """
        layers: list = []

        # iterate every node count except the last
        # (node count represent the in_nodes in the Layer class)
        for i, in_nodes in enumerate(nodes[:-1]):
            # get node count of next layer (the out nodes for the current layer)
            out_nodes = nodes[i + 1]

            layers.append(Layer(in_nodes, out_nodes))
            layers.append(ReLu())

        layers[-1] = Softmax()
        return layers

    def __forward_feed(self, x: np.ndarray) -> np.ndarray:
        """Forward propagate through all of the layers.

        Args:
            x: The input data to feed through the network.

        Returns:
            np.ndarray: The output of the last layer, i.e. the predicted labels.
        """
        # feed values through all of the layers,
        # with each layer getting the output of the previous layer as its input
        for layer in self.__layers:
            x = layer.forward(x)

        return x

    def __backpropagate(self, delta: np.ndarray, learning_rate: float) -> None:
        """Backpropagate through all network layers.

        Args:
            delta: The derivative of the loss w.r.t. the final output.
            learning_rate: The learning rate for gradient descent.
        """
        # backpropagate through layers, passing on delta to previous layers
        for layer in reversed(self.__layers):
            delta = layer.backward(delta, learning_rate)
            # since there is no use for delta, it is simply discarded

    def __get_batch(
            self, x: np.ndarray, y: np.ndarray, batch_size: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the batch to train on.

        Args:
            x: The input of the training data.
            y: The labels of the training data.
            batch_size: The size of the batch. Defualts to None. Full data is used if None.

        Returns:
            tuple: The input and output for the batch.
        """
        # return original data if no batch size is given or larger than the amount of batches
        if batch_size is None or batch_size >= x.shape[0]:
            return (x, y)

        # if batch size is given, shuffle data row-wise
        # the input and output is first added together, then shuffled,
        # to make sure the input and the output still align after shuffling
        z = np.concatenate((x, y), axis=1)
        np.random.shuffle(z)

        # split shuffled data into input and output again
        result = np.split(z, [x.shape[1], z.shape[1]], axis=1)

        # only use first n (batch size) rows of shuffled data
        return (result[0][:batch_size], result[1][:batch_size])

    def train(
            self, data: tuple[np.ndarray, np.ndarray],
            learning_rate: float, its: int, batch_size: int = None
    ) -> dict[str, list]:
        """Train the network on provided training data.

        Args:
            data: The input and labels for the training data.
            learning_rate: The learning rate used for gradient descent.
            its: The amount of training iterations.
            batch_size: The size of the batches used to train. Defaults to None.
            If None, then the full dataset is used.

        Returns:
            dict[str, list]: A history of the cost and accuracy of the model.
        """
        x, y = data

        cce = CCE()
        history: dict[str, list] = {"cost": [], "accuracy": []}

        print("Learning...")
        # training loop
        for i in range(1, its + 1):
            # batch input and output
            x_b, y_b = self.__get_batch(x, y, batch_size)

            # forward feed and backpropagate through the network
            y_pred = self.__forward_feed(x_b)
            self.__backpropagate(cce.delta(y_pred, y_b), learning_rate)

            # add stats to history
            loss: float = cce.cost(y_pred, y_b)
            acc: float = accuracy(y_pred, y_b)
            history["cost"].append(loss)
            history["accuracy"].append(acc)

            # training progress display
            bar_len = 50
            filled = int(bar_len * i / its)
            progress_bar = "#" * filled + " " * (bar_len - filled)

            print(f"\r[{progress_bar}] {i}/{its} | Batch accuracy: {acc}%", end="", flush=True)

        # compute and display accuracy with the full data
        y_pred = self.__forward_feed(x)
        print(f"\nLearning completed! Ending accuracy: {accuracy(y_pred, y)}%")

        # cache and return history
        self.__history = history
        return history

    def test(self, data: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        """Test the network with the provided testing data.

        Args:
            data: The input and labels of the testing data.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Predicted labels, actual labels & accuracy.
        """
        x, y = data

        pred = self.__forward_feed(x)
        print(f"Testing accuracy: {accuracy(pred, y)}%")
        return (pred, y, accuracy(pred, y))

    def save(self, name: str) -> None:
        """Save all of the network weights and biases.

        Args:
            name: The name under which it should be saved.
        """
        # folder of this file
        src_dir = os.path.dirname(os.path.abspath(__file__))
        # go back one step (into the project root), then into the data folder
        root_dir = os.path.dirname(src_dir)
        save_dir = os.path.join(root_dir, "data", "networks", name)

        params: dict[str, np.ndarray] = {}

        # iterate all Layer classes
        i = 0
        for layer in self.__layers[::2]:
            weights, biases = layer.get_params()
            params[f"w_{i}"] = weights
            params[f"b_{i}"] = biases
            i += 2

        # save architecture and all parameters
        np.savez(
                save_dir, allow_pickle=False,
                arch=self.__arch,
                cost_history=self.__history["cost"], acc_history=self.__history["accuracy"],
                **params
        )

    @classmethod
    def load(cls, name: str) -> tuple['Network', dict[str, list]]:
        """Load network from a saved file stat.

        Args:
            name: The name under which the network was saved.
        """
        # folder of this file
        src_dir = os.path.dirname(os.path.abspath(__file__))
        # go back one step (into the project root), then into the data folder
        root_dir = os.path.dirname(src_dir)
        nw_dir = os.path.join(root_dir, "data", "networks")
        saved_dir = os.path.join(nw_dir, f"{name}.npz")

        # make sure file exists
        existing_files = os.listdir(nw_dir)
        if f"{name}.npz" not in existing_files:
            raise FileNotFoundError(
                f"Network data does not exist in {nw_dir}. Missing: {f"{name}.npz"}\n"
                f"Try checking which files exist, perhaps the wrong name was entered."
            )

        # restore architecture and parameters
        with np.load(saved_dir) as data:
            arch = data["arch"]
            costs = data["cost_history"]
            acc = data["acc_history"]

            params: dict[str, np.ndarray] = {}

            for key in data.files:
                if key != "arch":
                    params[key] = data[key]

        # create network and load weights into it
        network = cls(*arch)
        i = 0
        for layer in network.__layers[::2]:
            w = params[f"w_{i}"]
            b = params[f"b_{i}"]
            layer.set_params(w, b)
            i += 2

        return (network, {"cost": costs, "accuracy": acc})
