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
    network.train(train_data, 30, 10, 0.5, 5)
     
    correct = 0
    error = 0
    for x, y in test_data:
        y_pred = torch.argmax(network.evaluate(x))
        if y_pred == y: 
            correct += 1 
        else: 
            error += 1
    print("Model accuracy: ", correct/(correct+error))
    
if __name__ == "__main__":
    main()