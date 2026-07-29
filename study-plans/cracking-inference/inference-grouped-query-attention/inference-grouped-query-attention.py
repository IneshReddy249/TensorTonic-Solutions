import torch

def grouped_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    batch, seq, d_model = hidden_states.shape
    d_k = d_model // num_query_heads
    group_size = num_query_heads // num_kv_heads

    query = hidden_states @ w_q
    key = hidden_states @ w_k
    value = hidden_states @ w_v

    query = query.view(batch, seq, num_query_heads, d_k).transpose(1, 2)
    key = key.view(batch, seq, num_kv_heads, d_k).transpose(1, 2)
    value = value.view(batch, seq, num_kv_heads, d_k).transpose(1, 2)

    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    if causal:
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=hidden_states.device), 
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    attn = torch.matmul(weights, value)
    concat = attn.transpose(1, 2).contiguous().view(batch, seq, d_model)

    return concat @ w_o