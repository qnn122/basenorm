import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, embbed_size):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(embbed_size, embbed_size) # Single linear layer
        torch.nn.init.eye_(self.linear.weight) # Linear layer weights initialization

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        x = self.linear(x)
        return x
