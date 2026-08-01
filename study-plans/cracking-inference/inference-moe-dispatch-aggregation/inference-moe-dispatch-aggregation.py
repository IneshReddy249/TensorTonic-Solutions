import torch


def sparse_moe_forward(
    token_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (num_tokens, d_model)
    """
    num_tokens, d_model = token_states.shape
    num_experts = router_logits.shape[-1]

    sorted_vals, sorted_idx = torch.sort(router_logits, dim=-1, descending=True, stable=True)
    expert_indices = sorted_idx[:, :top_k]
    routing_weights = torch.softmax(sorted_vals[:, :top_k], dim=-1)

    output = torch.zeros(num_tokens, d_model, dtype=token_states.dtype)

    for expert_id in range(num_experts):
        mask = expert_indices == expert_id
        if not mask.any():
            continue

        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x = token_states[token_idx]

        hidden = torch.relu(x @ w_in[expert_id])
        expert_out = hidden @ w_out[expert_id]

        weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        output.index_add_(0, token_idx, expert_out * weights)

    return output
