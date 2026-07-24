import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype= float)
    y = torch.tensor(y, dtype= float)

    if op == "add":
        return (x+y).tolist()
        
    elif op == "multiply":
        return (x * y).tolist()
        
    elif op == "matmul":
        return (x @ y).tolist()
        
    elif op == "power":
        return (x ** y).tolist()

    elif op == "max":
        return torch.maximum(x, y).tolist()
    pass