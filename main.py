from network import Simple_MLP, Optmized_MLP, load
from mnist import MNIST
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparams
    sizes = (784, 30, 10)
    epochs = 3
    mini_batch_size = 10
    eta = 0.5 
    lambd = 5
    
    network = Optmized_MLP(sizes, device)
    mnist = MNIST(device)
    
    (training_data, validation_data, test_data)= mnist.format_data()
    
    train_data = [ (x, y) for x, y in training_data ]
    val_data = [ (x, y) for x, y in validation_data ]
    test_data = [(x, y) for x, y in test_data]

    if isinstance(network, Simple_MLP): lambd = 0
    
    network.train(
            train_data, 
            epochs, 
            mini_batch_size, 
            eta, 
            lambd, 
            evaluation_data = val_data, 
            monitor_evaluation_accuracy=True, 
            monitor_evaluation_cost=True, 
            monitor_training_accuracy=True,
            monitor_training_cost=True)
    
    network.save(f"{network.__class__.__name__}.json")
    
    final_acurracy = network.accuracy(test_data)
    print(f"Final cost on test data: {network.total_cost(test_data, lambd)}")
    print(f"Final Acurracy test data: {final_acurracy}/{len(test_data)} = {final_acurracy/len(test_data)}")
    
if __name__ == "__main__":
    main()