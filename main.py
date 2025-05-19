from network import MLP
from mnist import MNIST
import torch

def main():
    sizes = (784, 16, 16, 10)
    network = MLP(sizes, None)
    mnist = MNIST(None)
    
    (training_data, validation_data, test_data)= mnist.format_data()
    
    train_data = [ (x, y) for x, y in training_data ]
    network.train(train_data, 10, 5, 0.5)
    
if __name__ == "__main__":
    main()