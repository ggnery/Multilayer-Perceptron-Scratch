from typing import Tuple, List
from .utils import sigmoid, sigmoid_derivative
import torch
import random
from abc import ABC, abstractmethod

class MLP(ABC):
    device: torch.device
    n_layers: int
    bias: List[torch.Tensor]
    weights: List[torch.Tensor]
    activations: List[torch.Tensor]
    zs:  List[torch.Tensor]
    
    def __init__(self, sizes: Tuple[int], device: torch.device | None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        
        self.device = device
        self.n_layers = len(sizes)
        self.activations = [torch.zeros(i) for i in sizes]
        self.zs = [torch.zeros(i) for i in sizes[1:]]
        
        """
        m = number of neurons in next layer (l) excluding input layer
        eg.: sizes (2, 3, 1) 
        b = [
            LAYER 2  
            [b1, b2, b3]
        ] 
        """
        self.bias = [ torch.rand(m, device = device) for m in sizes[1:] ] 
        
        """
        m = number of neurons in previous layer (l - 1)
        n = number of neurons in next layer (l)
        
        eg.: sizes (2, 3, 1) 
        w = [
            LAYER 1 <-> LAYER 2       LAYER 2 <-> LAYER 3           
            [                         [
                [w11, w12],                [w11],          
                [w21, w22],                [w21],
                [w31, w32]                 [w31]
            ] 3x2                     ] 3x1
        ]
        """
        self.weights = self.initialize_weights(sizes)
    
    @abstractmethod
    def initialize_weights(self, sizes: Tuple[int]) -> List[torch.Tensor]:
        pass
    
    def train(self, training_data: List[Tuple[torch.Tensor, torch.Tensor]], epochs: int, mini_batch_size: int, eta: float):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, eta)
            print(f"Epoch {j} complete")
    
    def evaluate(self, a):
        """Evaluate the output for some input"""
        for b_l, w_l in zip(self.bias, self.weights):
            a = sigmoid(torch.matmul(w_l, a) + b_l )
        return a
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Return the output of the network if ``a`` is input. Equation: a^l = sigmoid(w^l * a^(l-1) + b^(l-1))"""  
        self.activations[0] = a # save input layer activations
        for l,(w_l, b_l) in enumerate(zip(self.weights, self.bias)): # Supose that l start at 1 to fit the equation
            z_l = torch.matmul(w_l, self.activations[l]) + b_l 
            a_l = sigmoid(z_l) # σ(z^l)
            self.zs[l] = z_l # save z^l
            self.activations[l+1] = a_l # save new activations 
        
        return self.activations[-1] # Return last element in activations (activation outuput)
    
    def gradient_descent(self, mini_batch: List[Tuple[torch.Tensor, torch.Tensor]], eta: float):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        mean_delta_b = [torch.zeros(b.shape, device=self.device) for b in self.bias] 
        mean_delta_w = [torch.zeros(w.shape, device=self.device) for w in self.weights] 
        
        for x, y in mini_batch:
            self.forward(x)
            delta_w_x, delta_b_x = self.backprop(y)
            mean_delta_w = [nw+dnw for nw, dnw in zip(mean_delta_w, delta_w_x)] # ∑ δ^(x,l)
            mean_delta_b = [nb+dnb.squeeze() for nb, dnb in zip(mean_delta_b, delta_b_x)] # ∑(δ^(x,l) * (a^(x,l−1))^T)       

        self.weights = [w-(eta/len(mini_batch))* nw for w, nw in zip(self.weights, mean_delta_w)] # w^l → w^l − (η/m)*∑(δ^(x,l) * (a^(x,l−1))^T)
        self.bias = [b-(eta/len(mini_batch))* nb for b, nb in zip(self.bias, mean_delta_b)] # b^l→b^l − (η/m)* ∑δ^(x,l)
            
    
    def backprop(self, y: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return a tuple ``(delta_w, delta_b)`` representing the
        gradient for the cost function C_x.  ``delta_w`` and
        ``delta_b`` are layer-by-layer lists of tensors, similar
        to ``self.bias`` and ``self.weights``."""   
        error_L = MLP.cost_derivative(y, self.activations[-1]) * (sigmoid_derivative(self.zs[-1])) # Calculate error in last layer (L) with δ^L=∇aC ⊙ σ′(z^L).
        error_L = error_L.unsqueeze(1) # error_L by default is just an "array" so it has to be transformed into a matrix nx1

        delta_w = [torch.matmul(error_L, self.activations[-2].unsqueeze(1).transpose(1,0))] # Stores ∂C/∂w^l (the las element is δ^L * a^(L-1))
        errors = [error_L] # stores erros for all layers 

        for l in range(2, self.n_layers): # l = L-1, L-2, ..., 2
            error_l = torch.matmul(torch.transpose(self.weights[-l+1], 0, 1), errors[-l+1]) * sigmoid_derivative(self.zs[-l]).unsqueeze(1) # Calculate error in layer (l) with δ^l=((w^(l+1))^T * δ^(l+1)) ⊙ σ′(z^l)
            errors.insert(0, error_l) # insert δ^l in begining of errors list  
            delta_w.insert(0, torch.matmul(error_l, self.activations[-l-1].unsqueeze(1).transpose(1,0))) #  ∂C/∂w^l = δ^L *(a^l)^T 

        delta_b = errors # ∂C/∂b^l = δ^l  
        return (delta_w, delta_b)
    
    @staticmethod
    def cost_derivative(y: torch.Tensor, a_L: torch.Tensor) -> torch.Tensor:
        """Derivation of cost function for one train example x where this cost funciton is 
                                            n_L
        C_x = (1/2) * (y - a_l)^2 = (1/2) *  Σ (y_j - aL_j)^2
                                            j=0
        """
        return a_L - y
