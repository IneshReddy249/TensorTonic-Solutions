import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # 1. Define weight and bias as learnable parameters
        # Shape of weight: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Shape of bias: (out_features,)
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # 2. Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight using Kaiming Uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        # Compute bound for uniform bias initialization (standard PyTorch logic)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute y = x * W^T + b
        return x @ self.weight.t() + self.bias