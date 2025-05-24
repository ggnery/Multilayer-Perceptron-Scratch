from .mlp import MLP
from typing import Tuple, List
from .utils import sigmoid_derivative
import torch

class Simple_MLP(MLP):
    
    def initialize_weights(self, sizes: Tuple[int]) -> List[torch.Tensor]:
        return [ torch.randn(j, k, device= self.device) for k, j in zip(sizes[:-1], sizes[1:]) ]
    
    def delta(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        error = QuadraticCost.cost_derivative(x, y) * sigmoid_derivative(z) # Calculate error in last layer (L) with δ^L=∇aC ⊙ σ′(z^L).
        error = error.unsqueeze(1) # error by default is just an "array" so it has to be transformed into a matrix nx1
        return error

    def update_weights(self, n: int, m: int, eta: float, lambd: float, mean_delta_w: List[torch.Tensor]) -> List[torch.Tensor]:
        return [w-(eta/m) * nw for w, nw in zip(self.weights, mean_delta_w)] # w^l → w^l − (η/m)*∑(δ^(x,l) * (a^(x,l−1))^T)
        
    def update_bias(self, n: int, m: int, eta: float, mean_delta_b: List[torch.Tensor]) -> List[torch.Tensor]:
        return [b-(eta/m) * nb for b, nb in zip(self.bias, mean_delta_b)] # b^l→b^l − (η/m)* ∑δ^(x,l)

class QuadraticCost():
    @staticmethod
    def cost(a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5 * (y - a)^2
    
    def cost_derivative(a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Derivation of cost function for one train example x where this cost funciton is 
                                            n_L
        C_x = (1/2) * (y - a_l)^2 = (1/2) *  Σ (y_j - aL_j)^2
                                            j=0
        """
        return a - y