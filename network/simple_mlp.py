from .mlp import MLP
from typing import Tuple, List
import torch

class Simple_MLP(MLP):
    
    def initialize_weights(self, sizes: Tuple[int]) -> List[torch.Tensor]:
        return [ torch.randn(j, k, device= self.device) for k, j in zip(sizes[:-1], sizes[1:]) ]
    
    
