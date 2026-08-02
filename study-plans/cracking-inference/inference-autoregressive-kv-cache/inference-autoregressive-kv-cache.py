import torch
from typing import Tuple

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (outputs, key_cache, value_cache) for the full sequence, built incrementally
    """
    batch, seq, d_k = query.shape

    outputs = []
    key_cache = None
    value_cache = None

    for t in range(seq):

        new_k = key[:, t:t + 1, :]
        new_v = value[:, t:t + 1, :]

        if key_cache is None:
            key_cache = new_k
            value_cache = new_v

        else:
            key_cache = torch.cat([key_cache, new_k], dim=1)
            value_cache = torch.cat([value_cache, new_v], dim=1)

        q_t = query[:, t:t+1, :]
        scores = torch.matmul(q_t, key_cache.transpose(-2, -1)) / (d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out_t = torch.matmul(weights, value_cache)
        outputs.append(out_t)

    outputs = torch.cat(outputs, dim=1)
    return outputs, key_cache, value_cache
        
        
    
    
