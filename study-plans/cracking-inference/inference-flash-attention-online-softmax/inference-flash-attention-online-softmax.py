import torch


def flash_attention_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    batch, seq_q, d_k = query.shape
    _, seq_k, d_v = value.shape
    output = torch.zeros(batch, seq_q, d_v, dtype=query.dtype)

    def safe_exp(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff = x - ref
        return torch.where(torch.isneginf(ref), torch.zeros_like(diff), torch.exp(diff))

    for q_start in range(0, seq_q, query_block_size):
        q_end = min(q_start + query_block_size, seq_q)
        q_block = query[:, q_start:q_end, :]
        qb = q_end - q_start

        acc = torch.zeros(batch, qb, d_v, dtype=query.dtype)
        m_i = torch.full((batch, qb), float("-inf"), dtype=query.dtype)
        l_i = torch.zeros(batch, qb, dtype=query.dtype)

        for k_start in range(0, seq_k, key_block_size):
            k_end = min(k_start + key_block_size, seq_k)
            k_block = key[:, k_start:k_end, :]
            v_block = value[:, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / (d_k ** 0.5)

            if causal:
                q_idx = torch.arange(q_start, q_end).unsqueeze(1)
                k_idx = torch.arange(k_start, k_end).unsqueeze(0)
                block_mask = k_idx > q_idx
                scores = scores.masked_fill(block_mask, float("-inf"))

            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(m_i, block_max)

            alpha = safe_exp(m_i, new_max)
            p = safe_exp(scores, new_max.unsqueeze(-1))
            block_sum = p.sum(dim=-1)

            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_block)
            l_i = l_i * alpha + block_sum
            m_i = new_max

        output[:, q_start:q_end, :] = acc / l_i.unsqueeze(-1)

    return output
