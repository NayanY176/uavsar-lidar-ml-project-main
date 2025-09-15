import torch.nn as nn

class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(RegressionNN, self).__init__()
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        
        # Output layer
        # The output layer now takes input from the second hidden layer
        self.fc3 = nn.Linear(hidden_size2, 1)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))    # First hidden layer with ReLU activation
        
        x = self.relu(self.fc2(x))    # Second hidden layer with ReLU activation
        
        x = self.fc3(x)               # Output layer (no activation for regression)
        
        return x
