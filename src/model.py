"""File containing the logic of the network, represented as a class.

Usage example:

    network = Network(3, 2, 2)
    x_train = np.array([1, 0, 1])
    y_train = np.array([[1]])
    network.train(x_train, 0.01, y_train)
"""
import numpy as np
from src.loss import Loss
from src.layer import Layer
from src.activations import Sigmoid, Softmax

class Network:
    """A class representing the actual network.

    Pieces together the individual parts (layers, activations, loss) to get a running network.

    Attributes:
        __layers: The layers of the network, including activation layers.
    """
    def __init__(self, *nodes: tuple[int]) -> None:
        """Initialises instances using node amounts.

        Args:
            nodes: A tuple of node counts. Each node count represents the amount of nodes of one layer.
        """
        self.__layers = self.__init_layers(nodes)

    def __init_layers(self, nodes: tuple[int]) -> list:
        """Takes a tuple containing layer nodes and creates a list of actual layers.

        Args:
            nodes: Tuple of integers representing the layers and their node counts.

        Returns:
            list: List of layers representing the network.
        """
        layers: list = []

        # iterate every node count except the last (node count represent the in_nodes in the Layer class)
        for i, in_nodes in enumerate(nodes[:-1]):
            # get node count of next layer (the out nodes for the current layer)
            out_nodes = nodes[i + 1]

            layers.append(Layer(in_nodes, out_nodes))
            layers.append(Sigmoid())

        layers[-1] = Softmax()
        return layers

    def forward_feed(self, x: np.ndarray) -> np.ndarray:
        """Forward propagate through all of the layers.

        Args:
            x: The input data to feed through the network.

        Returns:
            np.ndarray: The output of the last layer, i.e. the predicted labels.
        """
        # feed values through all of the layers, with each layer getting the output of the previous layer as its input
        for layer in self.__layers:
            x = layer.forward(x)

        return x

    def backpropagate(self, delta: np.ndarray, learning_rate: float) -> None:
        """Backpropagate through all network layers.

        Args:
            delta: The derivative of the loss w.r.t. the final output.
            learning_rate: The learning rate for gradient descent.
        """
        # backpropagate through layers, passing on delta to previous layers
        for layer in reversed(self.__layers):
            delta = layer.backward(delta, learning_rate)
            # since there is no use for delta, it is simply discarded

    def __get_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int | None) -> tuple[np.ndarray, np.ndarray]:
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
        # the input and output is first added together, then shuffled, to make sure the input and the output still align
        z = np.concatenate((x, y), axis=1)
        np.random.shuffle(z)

        # split shuffled data into input and output again
        result = np.split(z, [x.shape[1], z.shape[1]], axis=1)

        # only use first n (batch size) rows of shuffled data
        return (result[0][:batch_size], result[1][:batch_size])

    def train(self, data: tuple[np.ndarray, np.ndarray], learning_rate: float, its: int, batch_size: int = None) -> dict[str, list]:
        """Train the network on provided training data.

        Args:
            data: The input and labels for the training data.
            learning_rate: The learning rate used for gradient descent.
            its: The amount of training iterations.
            batch_size: The size of the "mini" batches used to train. Defaults to None. If None, then the full dataset is used.

        Returns:
            dict[str, list]: A history of the cost and accuracy of the model.
        """
        x, y = data

        l = Loss()
        history: dict[str, list] = {"cost": [], "accuracy": []}

        print("Learning...")
        # training loop
        for i in range(1, its + 1):
            # batch input and output
            x_b, y_b = self.__get_batch(x, y, batch_size)

            # forward feed and backpropagate through the network
            y_pred = self.forward_feed(x_b)
            self.backpropagate(l.delta(y_pred, y_b), learning_rate)

            # add stats to history
            loss: float = l.cost(y_pred, y_b)
            accuracy: float = l.accuracy(y_pred, y_b)
            history["cost"].append(loss)
            history["accuracy"].append(accuracy)

            # training progress display
            bar_len = 50
            filled = int(bar_len * i / its)
            progress_bar = "#" * filled + " " * (bar_len - filled)

            print(f"\r[{progress_bar}] {i}/{its} | Batch accuracy: {accuracy}%", end="", flush=True)

        y_pred = self.forward_feed(x)
        print(f"\nLearning completed! Ending accuracy: {l.accuracy(y_pred, y)}%")
        return history

    def test(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test the network with the provided testing data.

        Args:
            data: The input and labels of the testing data.
        """
        x, y = data

        l = Loss()

        pred = self.forward_feed(x)
        print(f"Testing accuracy: {l.accuracy(pred, y)}%")

