from .mlp import MLP
from typing import Tuple, List
import torch
import math

class Optmized_MLP(MLP):
    
    def initialize_weights(self, sizes: Tuple[int]) -> List[torch.Tensor]:
        return [ torch.randn(j, k, device= self.device)/math.sqrt(k) for k, j in zip(sizes[:-1], sizes[1:]) ]