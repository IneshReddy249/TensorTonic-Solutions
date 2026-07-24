import torch

def activate(x, method="relu"):
    
     x = torch.tensor(x, dtype=float)
     if method == "relu":
         return torch.clamp(x, min = 0).tolist()
         
     elif  method == "sigmoid":
         return (1.0 / (1.0 + torch.exp(-x))).tolist()
         
     elif method == "tanh":
         return torch.tanh(x).tolist()

     elif method == "leaky_relu":
        return torch.where(x > 0, x, 0.01 * x).tolist()
