import torch

def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Sigmoid funciton"""
    return 1.0 / (1.0 + torch.exp(-z))

def sigmoid_derivative(z: torch.Tensor) -> torch.Tensor:
    """Derivative of sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))

def vectorized_result(j: torch.Tensor):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = torch.zeros(10)
        e[j] = 1.0
        return e