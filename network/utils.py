import torch

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Sigmoid funciton"""
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_derivative(z: torch.Tensor) -> torch.Tensor:
    """Derivative of sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))