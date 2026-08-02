import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns: torch.int64 tensor of shape (4,) ordered [MHA, MQA, GQA, MLA]
    """
    total_tokens = batch_size * seq_len * num_layers

    mha_bytes = total_tokens * 2 * num_query_heads * head_dim * bytes_per_element

    gqa_bytes = total_tokens * 2 * gqa_kv_heads * head_dim * bytes_per_element

    mqa_bytes = total_tokens * 2 * 1 * head_dim * bytes_per_element

    mla_bytes = total_tokens * (mla_latent_dim + mla_rotary_key_dim) * bytes_per_element

    return torch.tensor([mha_bytes, mqa_bytes, gqa_bytes, mla_bytes], dtype=torch.int64)

    
    
    pass
