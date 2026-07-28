import math
import torch


def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """Returns:

    output tensor of shape (batch, seq, d_model)
    """
    # Convert inputs to float32 tensors
    hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
    w_q = torch.as_tensor(w_q, dtype=torch.float32)
    w_k = torch.as_tensor(w_k, dtype=torch.float32)
    w_v = torch.as_tensor(w_v, dtype=torch.float32)
    w_o = torch.as_tensor(w_o, dtype=torch.float32)

    batch, seq, d_model = hidden_states.shape
    d_k = d_model // num_query_heads

    # 1. Linear Projections
    # Q shape: (batch, seq, d_model)
    # K shape: (batch, seq, d_k)
    # V shape: (batch, seq, d_k)
    q = torch.matmul(hidden_states, w_q)
    k = torch.matmul(hidden_states, w_k)
    v = torch.matmul(hidden_states, w_v)

    # 2. Reshape Q into multi-head format: (batch, num_query_heads, seq, d_k)
    q = q.view(batch, seq, num_query_heads, d_k).transpose(1, 2)

    # 3. Add head dimension to K and V for broadcasting across query heads:
    # K, V shape: (batch, 1, seq, d_k)
    k = k.unsqueeze(1)
    v = v.unsqueeze(1)

    # 4. Compute Attention Scores: Q @ K^T / sqrt(d_k)
    # Scores shape: (batch, num_query_heads, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 5. Apply Causal Mask if required
    if causal:
        mask = torch.triu(
            torch.full((seq, seq), float("-inf"), device=scores.device),
            diagonal=1,
        )
        scores = scores + mask

    # 6. Softmax and Attention Output Computation
    attn_weights = torch.softmax(scores, dim=-1)
    # Output shape: (batch, num_query_heads, seq, d_k)
    attn_output = torch.matmul(attn_weights, v)

    # 7. Concatenate heads back to (batch, seq, d_model)
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(batch, seq, d_model)
    )

    # 8. Final Output Projection
    output = torch.matmul(attn_output, w_o)

    return output