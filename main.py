# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: notebooks//ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
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
print("MNIST data loaded:")
print(f"Training shapes: {x_train.shape} & {y_train.shape}")
print(f"Testing shapes: {x_test.shape} & {y_test.shape}")
print()

# %% [markdown]
# ## Initialise the network

# %%
network = Network(784, 128, 64, 10)
# use this line instead to load a pretrained network
# network, history = Network.load("readme")

# %% [markdown]
# ## Training the network

# %%
history = network.train((x_train, y_train), 0.01, 15000, batch_size=256)
print()

# %% [markdown]
# ## Testing the network

# %%
pred, act, acc = network.test((x_test, y_test))

# %% [markdown]
# ## Saving the model and its results

# %%
name: str = "arch_784_128_64_10-lr_0.01-its_15000-bs_256"
network.save(name)
save_cost(history["cost"], name)
save_accuracy(history["accuracy"], acc, name)
save_confusion_matrix(pred, act, name)
save_classified(x_test, pred, act, name)
