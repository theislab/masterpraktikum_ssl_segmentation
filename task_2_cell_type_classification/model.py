import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        # Initialize weights using Xavier/Glorot initialization and biases to zero
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return F.softmax(x, dim=1)  # Apply softmax to get probabilities

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        # Define the hidden layer
        self.hidden_layer = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights using Xavier/Glorot initialization and biases to zero
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0)
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.constant_(self.hidden_layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.input_layer(x))  # Activation function for hidden layer
        x = self.hidden_layer(x)
        return F.softmax(x, dim=1)  # Apply softmax to get probabilities
