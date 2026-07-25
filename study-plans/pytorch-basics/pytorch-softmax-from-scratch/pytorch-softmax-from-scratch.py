import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    logits = torch.tensor(logits, dtype=float)
    max_values = torch.max(logits, dim=1, keepdim=True).values
    exps = torch.exp(logits - max_values)
    return exps /  exps.sum(dim=1, keepdim=True) 


    
    
    pass
