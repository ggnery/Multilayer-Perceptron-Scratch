# 🧠 Multilayer Perceptron from Scratch

A comprehensive PyTorch implementation of Multilayer Perceptron (MLP) neural networks built from scratch. This project demonstrates the fundamental concepts of neural networks including forward propagation, backpropagation, gradient descent optimization, and different cost functions.

## ✨ Features

### 🏗️ Network Architectures
- 🔹 **Simple MLP**: Basic implementation using quadratic cost function with standard weight initialization
- 🔹 **Optimized MLP**: Advanced implementation with improved weight initialization and cross-entropy cost function
- 🔹 **Configurable Architecture**: Support for arbitrary layer sizes and depths
- 🔹 **GPU Support**: CUDA acceleration when available

### 🎯 Training Features
- 🚀 **Mini-batch Stochastic Gradient Descent**: Efficient training with configurable batch sizes
- 🛡️ **L2 Regularization**: Prevent overfitting with configurable regularization parameter
- 📊 **Multiple Cost Functions**: 
  - Quadratic Cost (Mean Squared Error)
  - Cross-Entropy Cost (for better classification performance)
- 📈 **Comprehensive Monitoring**: Track training/validation accuracy and cost during training
- 💾 **Model Persistence**: Save and load trained models

### 📚 Dataset Support
- 🔢 **MNIST Dataset**: Built-in loader with proper preprocessing
- 🔄 **Flexible Data Format**: Support for custom datasets with proper formatting

## 📋 Requirements

Create a requirements.txt file or install the following:

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
```

## 🚀 Installation

1. 📥 Clone the repository:
```bash
git clone <repository-url>
cd Multilayer-Perceptron-Scratch
```

2. 📦 Install dependencies:
```bash
pip install torch torchvision numpy
```

3. 📊 Download MNIST dataset (automatic on first run):
```bash
python main.py
```

## 💻 Usage

### ⚡ Basic Training

Train an optimized MLP on MNIST with default hyperparameters:

```python
python main.py
```

### ⚙️ Custom Configuration

```python
from network import Simple_MLP, Optmized_MLP, load
from mnist import MNIST
import torch

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure network architecture
sizes = (784, 128, 64, 10)  # Input layer, hidden layers, output layer

# Create network (choose one)
network = Optmized_MLP(sizes, device)  # Recommended for better performance
# network = Simple_MLP(sizes, device)   # Basic implementation

# Load and format MNIST data
mnist = MNIST(device)
training_data, validation_data, test_data = mnist.format_data()

# Configure training parameters
epochs = 10
mini_batch_size = 16
learning_rate = 0.1
regularization = 5.0  # L2 regularization parameter

# Train the network
network.train(
    list(training_data),
    epochs=epochs,
    mini_batch_size=mini_batch_size,
    eta=learning_rate,
    lambd=regularization,
    evaluation_data=list(validation_data),
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True
)

# Save the trained model
network.save("my_model.json")
```

### 📁 Loading a Trained Model

```python
from network import load, Optmized_MLP

# Load a saved model
network = load("Optmized_MLP.json", device, Optmized_MLP)

# Evaluate on new data
accuracy = network.accuracy(test_data)
print(f"Test accuracy: {accuracy}")
```

## 📂 Project Structure

```
Multilayer-Perceptron-Scratch/
├── 🧠 network/
│   ├── __init__.py           # Package initialization and exports
│   ├── mlp.py               # Abstract base MLP class with core functionality
│   ├── simple_mlp.py        # Basic MLP with quadratic cost
│   ├── optimized_mlp.py     # Advanced MLP with cross-entropy cost
│   └── utils.py             # Activation functions and utilities
├── 🔢 mnist/
│   ├── __init__.py          # Package initialization
│   └── mnist_loader.py      # MNIST dataset loading and preprocessing
├── 📊 data/                    # Directory for datasets (created automatically)
├── 🚀 main.py                  # Main training script
├── 📄 LICENSE                  # MIT License
└── 📖 README.md               # This file
```

## 🔬 Algorithm Details

### ➡️ Forward Propagation
The network computes activations using:
$$a^l = \sigma(W^l \cdot a^{l-1} + b^l)$$
where $\sigma$ is the sigmoid activation function.

### ⬅️ Backpropagation
Gradients are computed using:
- 🎯 **Output layer error**: $\delta^L = \nabla_a C \odot \sigma'(z^L)$
- 🔄 **Hidden layer error**: $\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$
- ⚖️ **Weight gradients**: $\frac{\partial C}{\partial W^l} = \delta^l (a^{l-1})^T$
- 📐 **Bias gradients**: $\frac{\partial C}{\partial b^l} = \delta^l$

### 📊 Cost Functions

**🔸 Quadratic Cost (Simple MLP)**:
$$C = \frac{1}{2n} \sum \|y - a\|^2$$

**🔹 Cross-Entropy Cost (Optimized MLP)**:
$$C = -\frac{1}{n} \sum [y \ln(a) + (1-y) \ln(1-a)]$$

### 🛡️ Regularization
L2 regularization is applied to weights:
$$C_{regularized} = C + \frac{\lambda}{2n} \sum \|W\|^2$$

## 🆚 Key Differences Between Implementations

| Feature | 🔸 Simple MLP | 🔹 Optimized MLP |
|---------|------------|---------------|
| 🎲 Weight Initialization | Normal(0,1) | Normal(0,1/√n) |
| 📊 Cost Function | Quadratic | Cross-Entropy |
| ⚡ Learning Speed | Slower | Faster |
| 🎯 Performance | Good | Better |

## 📈 Performance

On MNIST dataset with default hyperparameters:
- 🔸 **Simple MLP**: ~93% accuracy on test set
- 🔹 **Optimized MLP**: ~95%+ accuracy on test set
- ⏱️ **Training Time**: ~2-3 minutes on CPU, <1 minute on GPU

## 🎛️ Hyperparameter Guidelines

- 📏 **Learning Rate (eta)**: Start with 0.1-0.5, reduce if loss oscillates
- 📦 **Mini-batch Size**: 10-32 for small datasets, 32-128 for larger ones
- 🛡️ **Regularization (lambda)**: 0.1-10, higher values for more regularization
- 🏗️ **Hidden Layers**: Start with 1-2 layers, 30-128 neurons each

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 🎉