from typing import Tuple, List, Type
from .utils import sigmoid, sigmoid_derivative, vectorized_result
import torch
import random
import json
from abc import ABC, abstractmethod

class MLP(ABC):
    device: torch.device
    sizes: Tuple[int]
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
        self.sizes = sizes
        self.n_layers = len(sizes)
        self.activations = [torch.zeros(i, device=self.device) for i in sizes]
        self.zs = [torch.zeros(i, device=self.device) for i in sizes[1:]]
        
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
    
    @abstractmethod
    def delta(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def cost(self, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
    
    def train(self, 
              training_data: List[Tuple[torch.Tensor, torch.Tensor]], 
              epochs: int, 
              mini_batch_size: int, 
              eta: float,
              lambd: float = 0.0,
              evaluation_data: List[Tuple[torch.Tensor, torch.Tensor]] = None,
              monitor_evaluation_cost=False,
              monitor_evaluation_accuracy=False,
              monitor_training_cost=False,
              monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [ training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, eta, lambd, n)
            
            print("=========================")
            print(f"Epoch {j} complete")
            print("=========================")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lambd)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n} = {accuracy/n}")
            
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lambd, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy} / {n_data} = {accuracy/n_data}")
            print("=========================")
        
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
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
    
    def gradient_descent(self, mini_batch: List[Tuple[torch.Tensor, torch.Tensor]], eta: float, lambd: float, n: int):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        mean_delta_b = [torch.zeros(b.shape, device=self.device) for b in self.bias] 
        mean_delta_w = [torch.zeros(w.shape, device=self.device) for w in self.weights] 
        
        for x, y in mini_batch:
            self.forward(x)
            delta_w_x, delta_b_x = self.backprop(y)
            mean_delta_w = [nw+dnw for nw, dnw in zip(mean_delta_w, delta_w_x)] # ∑(δ^(x,l) * (a^(x,l−1))^T)    
            mean_delta_b = [nb+dnb.squeeze() for nb, dnb in zip(mean_delta_b, delta_b_x)] # ∑ δ^(x,l)

        self.weights = [(1-(eta*lambd)/n)*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, mean_delta_w)] # w^l→(1-η*λ/n)w^l − (η/m)* ∑∂C/∂w^l
        self.bias = [b-(eta/len(mini_batch))* nb for b, nb in zip(self.bias, mean_delta_b)] # b^l→b^l − (η/m)* ∑δ^(x,l)
            
    
    def backprop(self, y: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return a tuple ``(delta_w, delta_b)`` representing the
        gradient for the cost function C_x.  ``delta_w`` and
        ``delta_b`` are layer-by-layer lists of tensors, similar
        to ``self.bias`` and ``self.weights``."""   
        error_L = self.delta(self.activations[-1], y, self.zs[-1])
        
        delta_w = [torch.matmul(error_L, self.activations[-2].unsqueeze(1).transpose(1,0))] # Stores ∂C/∂w^l (the las element is δ^L * a^(L-1))
        errors = [error_L] # stores erros for all layers 

        for l in range(2, self.n_layers): # l = L-1, L-2, ..., 2
            error_l = torch.matmul(torch.transpose(self.weights[-l+1], 0, 1), errors[-l+1]) * sigmoid_derivative(self.zs[-l]).unsqueeze(1) # Calculate error in layer (l) with δ^l=((w^(l+1))^T * δ^(l+1)) ⊙ σ′(z^l)
            errors.insert(0, error_l) # insert δ^l in begining of errors list  
            delta_w.insert(0, torch.matmul(error_l, self.activations[-l-1].unsqueeze(1).transpose(1,0))) #  ∂C/∂w^l = δ^l *(a^l)^T 

        delta_b = errors # ∂C/∂b^l = δ^l  
        return (delta_w, delta_b)
    
    def accuracy(self, data: List[Tuple[torch.Tensor, torch.Tensor]], convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(torch.argmax(self.evaluate(x)), torch.argmax(y))
                       for (x, y) in data]
        else:
            results = [(torch.argmax(self.evaluate(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data: List[Tuple[torch.Tensor, torch.Tensor]], lmbda: float, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.evaluate(x)
            if convert: y = vectorized_result(y)
            cost += self.cost(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            torch.norm(w)**2 for w in self.weights)
        return cost
    
    def save(self, filename: str):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.bias],
                "activations": [a.tolist() for a in self.activations],
                "zs": [z.tolist() for z in self.zs]
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    

def load(filename: str, device: torch.device | None, clazz: Type[MLP]):
    """Load a neural network from the file ``filename``.  Returns an instance of Network."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    
    mlp = clazz(data["sizes"], device)
    mlp.weights = [torch.tensor(w) for w in data["weights"]]
    mlp.biases = [torch.tensor(b) for b in data["biases"]]
    mlp.activations = [torch.tensor(a) for a in data["activations"]]
    mlp.zs = [torch.tensor(z) for z in data["zs"]]
    
    return mlp