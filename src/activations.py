"""File containing the logic of the activation functions, represented as classes.

Usage example:

    sigmoid = Sigmoid(3, 1)
    input = np.array([1, 0, 1])
    output = layer.forward(input)
"""
import numpy as np

class Sigmoid:
    """A class representing the sigmoid activation in a neural network.

    In this code implementation, activation functions are logically implemented as a seperate layer in the network.
    Thus, activation functions have the same public functions as the Layer class.

    Attributes:
        __x: Caches the input of the activation function (needed for backpropagation).
    """
    def __init__(self) -> None:
        """Initialises instances.
        """
        self.__x: np.ndarray = None

    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function, squishes any value into the range of 0-1.

        Args:
            x: The input to squish.

        Returns:
            np.ndarray: The squished result.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the sigmoid and forward it to the next layer.

        Args:
            x: The input from the previous layer.

        Returns:
            np.ndarray: The input the upcoming layer should recieve.
        """
        # store input for gradient computation
        self.__x = x

        return self.__sigmoid(x)

    def backward(self, delta: np.ndarray, _) -> np.ndarray:
        """Compute the new delta and give it to the previous Layer.

        Args:
            delta: The derivative of the loss w.r.t. the output of this layer.
            _: A placeholder to be able to use this class the same way as the Layer class.

        Returns:
            np.ndarray: Delta, the derivative of the loss w.r.t. the output of the previous layer.
        """
        # add derivative of the sigmoid to delta
        sigmoid = self.__sigmoid(self.__x)
        return delta * (sigmoid * (1 - sigmoid))

class Softmax:
    """A class representing the softmax activation in a neural network.

    In this code implementation, activation functions are logically implemented as a seperate layer in the network.
    Thus, activation functions have the same public functions as the Layer class.
    """
    def __softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function. Squish any values into the range of 0-1, representing a probability. So sum of these = 1.

        Args:
            x: The input to squish to probabilities.

        Returns:
            np.ndarray: The squished result.
        """
        # the input is normalised (subtract each value by the maximum value) to avoid numerical overflow in the softmax function
        # this works since the softmax function is shift-invariant
        x = x - np.max(x, axis=1, keepdims=True) # shape (batches, nodes)

        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax and forward it to the next layer.

        Args:
            x: The input from the previous layer.

        Returns:
            np.ndarray: The input for the upcoming Layer.
        """
        return self.__softmax(x)


    def backward(self, delta: np.ndarray, _) -> np.ndarray:
        """Pass on delta to the previous Layer.

        Since our softmax layer is always the last layer in our network, its delta is the derivative of the loss
        w.r.t. the output of the network, which is calculated in the Loss class. This is done so that this class can
        be called in the same way as the Layer class.

        Args:
            delta: The derivative of the loss w.r.t. the output of this layer.
            _: A placeholder to be able to use this class the same way as the Layer class.

        Returns:
            np.ndarray: Delta, the derivative of the loss w.r.t. the output of the previous layer.
        """
        # simply pass on delta recieved from the Loss class
        return delta
