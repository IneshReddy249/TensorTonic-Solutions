import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    mean = X.mean(dim=0)
    var = X.var(dim=0, unbiased=False)
    X_norm = (X - mean) / torch.sqrt(var + eps)
    return gamma * X_norm + beta
