"""This is the main entry point file.

See also notebooks/main.ipynb for a jupyter notebook version.
"""

# %%
from src.loading import load_mnist
from src.model import Network
from src.visualisations import save_cost, save_accuracy, save_confusion_matrix, save_classified

# %% [markdown]
# ## Load MNIST dataset
# Be sure to have downloaded the raw data using data/loader.py

# %%
(x_train, y_train), (x_test, y_test) = load_mnist()
print("MNIST data loaded")

# %% [markdown]
# ## Initialise the network

# %%
network = Network(784, 512, 512, 10)

# to load an already trained network
# network, history = Network.load("readme")

# %% [markdown]
# ## Training the network

# %%
history = network.train((x_train, y_train), 0.1, 100, batch_size=128)
print()

# %% [markdown]
# ## Testing the network

# %%
pred, act, acc = network.test((x_test, y_test))

# %% [markdown]
# ## Saving the model and its results

# %%
name: str = "readme"
network.save(name)
save_cost(history["cost"], name)
save_accuracy(history["accuracy"], acc, name)
save_confusion_matrix(pred, act, name)
save_classified(x_test, pred, act, name)
