import torch
from typing import Tuple

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (expert_indices, routing_weights), each of shape (num_tokens, top_k)
    """

    sorted_logits, sorted_index = torch.sort(router_logits, dim=-1, descending=True, stable=True)
    top_logits = sorted_logits[:, :top_k]
    expert_indices = sorted_index[:, :top_k]
    routing_weights = torch.softmax(top_logits, dim=-1)
    return expert_indices, routing_weights
    pass
