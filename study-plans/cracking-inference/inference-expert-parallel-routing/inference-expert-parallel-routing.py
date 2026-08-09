import torch
from typing import Tuple

def prepare_expert_parallel_dispatch(
    selected_expert_ids: torch.Tensor,
    expert_to_rank: torch.Tensor,
    routing_weights: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    selected_expert_ids, routing_weights: shape (num_tokens, k)
    expert_to_rank: shape (num_experts,)
    Returns: (dispatch_permutation, per_rank_send_counts, inverse_permutation,
              reordered_expert_ids, reordered_weights), all flattened to (num_tokens * k,)
              except per_rank_send_counts which has shape (world_size,).
    """
    flat_expert_ids = selected_expert_ids.flatten()
    flat_weights = routing_weights.flatten()

    dest_ranks = expert_to_rank[flat_expert_ids]

    dispatch_permutation = torch.argsort(dest_ranks, stable=True)

    per_rank_send_counts = torch.bincount(dest_ranks, minlength=world_size)

    reordered_expert_ids = flat_expert_ids[dispatch_permutation]
    reordered_weights = flat_weights[dispatch_permutation]

    inverse_permutation = torch.argsort(dispatch_permutation)

    return (
        dispatch_permutation, 
        per_rank_send_counts, 
        inverse_permutation, 
        reordered_expert_ids, 
        reordered_weights
    )
    

    
    pass
