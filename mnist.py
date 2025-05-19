from network import MLP
import torch

def main():
    sizes = (784, 16, 16, 10)
    network = MLP(sizes, None)

    train_data = [ (torch.randn(sizes[0]), torch.randn(sizes[-1])) for i in range(100) ]
    network.train(train_data, 100, 5, 0.5)
    
    result = network.forward(torch.randn(sizes[0]))
    print(result)
    
if __name__ == "__main__":
    main()