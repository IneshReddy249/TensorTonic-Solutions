import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        
    