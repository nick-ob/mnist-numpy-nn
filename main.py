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
from src.visualisations import save_cost, save_accuracy, save_confusion_matrix

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
network = Network(x_train.shape[1], 64, 32, y_train.shape[1])

# %% [markdown]
# ## Training the network

# %%
history = network.train((x_train, y_train), 0.1, 50, batch_size=512)
print()

# %% [markdown]
# ## Testing the network

# %%
pred, act, acc = network.test((x_test, y_test))

# %% [markdown]
# ## Saving the results

# %%
# save_cost(history["cost"])
# save_accuracy(history["accuracy"])
# save_confusion_matrix(pred, act)
