import torch
from typing import Tuple

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 1. Compress hidden states into the low-rank latent tensor 'c'
    latent = torch.matmul(hidden_states, w_down)
    
    # 2. Compute the query states from hidden states
    Q = torch.matmul(hidden_states, w_q)
    
    # 3. Reconstruct keys and values from the latent tensor
    K = torch.matmul(latent, w_up_k)
    V = torch.matmul(latent, w_up_v)
    
    # 4. Reshape and transpose Q, K, V for multi-head attention
    batch_size, seq_len, d_model = hidden_states.shape
    d_k = d_model // num_heads
    
    # Shape: (batch, seq, num_heads, d_k) -> permute to (batch, num_heads, seq, d_k)
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    
    # 5. Compute scaled dot-product attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 6. Apply causal mask if requested
    if causal:
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
        scores = scores + mask
        
    # 7. Softmax normalization over the last dimension
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 8. Compute attention output values
    attn_output = torch.matmul(attn_weights, V)
    
    # 9. Combine heads back together and project through w_o
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    output = torch.matmul(attn_output, w_o)
    
    return output, latent