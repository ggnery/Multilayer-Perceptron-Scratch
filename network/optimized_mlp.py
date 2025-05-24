from .mlp import MLP
from typing import Tuple, List
import torch
from .utils import sigmoid_derivative
import math

class Optmized_MLP(MLP):
    
    def initialize_weights(self, sizes: Tuple[int]) -> List[torch.Tensor]:
        return [ torch.randn(j, k, device= self.device)/math.sqrt(k) for k, j in zip(sizes[:-1], sizes[1:]) ]
    
    def delta(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        error = x - y # Calculate error in last layer (L) with δ^L=∇aC ⊙ σ′(z^L).
        error = error.unsqueeze(1) # error by default is just an "array" so it has to be transformed into a matrix nx1
        return error
    
    
class CrossEntropy():

    def cost(a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        C_x=-[yln(a)+(1-y)ln(1-a)]
        """
        return torch.matmul(-y,  torch.log(a)) + torch.matmul(y - 1, torch.log(1 - a)) 
    
    def cost_derivative(a, y):
        return (a - y) / (a*(1 - a)) 