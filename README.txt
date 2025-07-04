House Price Prediction with PyTorch
A simple neural network implementation using PyTorch to predict house prices based on square footage. This project demonstrates fundamental deep learning concepts including linear regression, gradient descent, and model training using a single linear layer.
Project Overview
This project implements linear regression using PyTorch's neural network framework. The model learns to predict house prices from square footage data through gradient descent optimization. The implementation includes a custom neural network class, training loop, and achieves approximately 98% accuracy on test data with predictions within 2% of actual values.
Files

model.py - Neural network definition with linear layer
data.py - Data generation utilities
main.py - Training script with optimization loop
requirements.txt - Python dependencies

Usage
Install dependencies: pip install torch matplotlib numpy
Run training: python main.py
The model learns the relationship between square footage and price through 1000 training iterations using SGD optimization with MSE loss.