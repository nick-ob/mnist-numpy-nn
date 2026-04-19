# 🕸️ MNIST Neural Network from Scratch (NumPy)

🚧 Work in progress 🚧

## About

This project implements a neural network from scratch using only NumPy
to classify handwritten digits from the MNIST dataset.

The goal is to understand the internal mechanics of neural networks,
including forward propagation, backpropagation, and gradient-based optimization.

## Current Results

### Training Loss

![Training Loss](results/readme/cost_over_epochs.png)

The model shows a clear decrease in training loss over epochs,
indicating that it is successfully learning from the data.

Further visualisation can be found [here](results/).

## Status

- Core neural network implementation: ✔ complete
- MNIST data pipeline: ✔ complete
- Training loop: ✔ complete
- First tests: ✔ complete
- Visualization: 🚧 in progress
- Weight saving/loading: ✔ complete
- Code cleanup & refactoring: 🚧 in progress

## What I’m focusing on next

- Add confusion matrix visualization
- Save and reload trained model weights

## Tech Stack

- Python
- NumPy
- Seaborn
- Matplotlib

## Notes

This project avoids high-level ML frameworks to better understand
the underlying principles of neural networks.
