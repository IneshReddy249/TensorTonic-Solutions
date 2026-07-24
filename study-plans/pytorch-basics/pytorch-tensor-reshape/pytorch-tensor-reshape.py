import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype= float)
    if op == "flatten":
        return (x.flatten()).tolist()
        
    elif op == "squeeze":
        return (x.squeeze()).tolist()

    elif op == "transpose":
        return (x.T).tolist()
    pass
