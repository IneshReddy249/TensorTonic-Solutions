import torch


def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    # Ensure inputs are tensors
    hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
    w_q = torch.as_tensor(w_q, dtype=torch.float32)
    w_k = torch.as_tensor(w_k, dtype=torch.float32)
    w_v = torch.as_tensor(w_v, dtype=torch.float32)
    w_o = torch.as_tensor(w_o, dtype=torch.float32)

    batch, seq, d_model = hidden_states.shape
    d_k = d_model // num_heads

    query = hidden_states @ w_q
    key = hidden_states @ w_k
    value = hidden_states @ w_v

    query = query.view(batch, seq, num_heads, d_k).transpose(1, 2)
    key = key.view(batch, seq, num_heads, d_k).transpose(1, 2)
    value = value.view(batch, seq, num_heads, d_k).transpose(1, 2)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
    if causal:
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attn = torch.matmul(weights, value)

    concat = attn.transpose(1, 2).contiguous().view(batch, seq, d_model)
    return concat @ w_o