# Multilayer Perceptron from Scratch

A PyTorch implementation of a Multilayer Perceptron (MLP) neural network built from scratch. This project demonstrates the fundamentals of neural networks including forward propagation, backpropagation, and gradient descent optimization.

## Features

- Custom MLP implementation with configurable layer architecture
- Stochastic gradient descent with mini-batch optimization
- MNIST dataset loading and preprocessing
- Neural network training and evaluation

## Requirements

- Python 3.x
- PyTorch
- torchvision (for MNIST dataset loading)

## Usage

```python
# Train and evaluate the network on MNIST
python main.py
```

## Project Structure

- `network/` - Contains the MLP implementation
  - `mlp.py` - Core MLP class with forward and backward propagation
  - `utils.py` - Helper functions for activation functions
- `mnist/` - MNIST dataset loading and preprocessing
- `data/` - Directory for storing datasets

## How It Works

The implementation follows the standard neural network architecture:
1. Forward propagation: Computing activations through the network
2. Cost function calculation: Measuring the error of predictions
3. Backpropagation: Computing gradients for weights and biases
4. Gradient descent: Updating weights to minimize error

## License

This project is licensed under the MIT License - see the LICENSE file for details.