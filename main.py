from network import Simple_MLP, Optmized_MLP
from mnist import MNIST
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sizes = (784, 30, 10)
    network = Simple_MLP(sizes, device)
    mnist = MNIST(device)
    
    (training_data, validation_data, test_data)= mnist.format_data()
    
    train_data = [ (x, y) for x, y in training_data ]
    val_data = [ (x, y) for x, y in validation_data ]

    network.train(train_data, 30, 10, 0.5, 5, 
                  evaluation_data = val_data, 
                  monitor_evaluation_accuracy=True, 
                  monitor_evaluation_cost=True, 
                  monitor_training_accuracy=True,
                  monitor_training_cost=True)
    
if __name__ == "__main__":
    main()