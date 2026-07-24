import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    y = (x ** 3 + 2 * x).sum()
    y.backward()
    return x.grad.tolist()
