import torch
from typing import Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    dk = query.size(-1)

    
    scores = torch.matmul(query, key.transpose(-2, -1)) / (dk ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))


    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)

    
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    pass
