from network import MLP
import torch

def main():
    network = MLP((2,3,1), None)

    train_data = [
        (torch.tensor([1,2], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)), 
        (torch.tensor([2,1], dtype=torch.float32), torch.tensor([2], dtype=torch.float32)),
        (torch.tensor([2,2], dtype=torch.float32), torch.tensor([2], dtype=torch.float32))]
    
    network.train(train_data, 1, 2, 0.5)
    network.forward((torch.tensor([1,1]), torch.tensor([1])))
    
if __name__ == "__main__":
    main()