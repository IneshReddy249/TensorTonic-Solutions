import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    uniform_draws: torch.Tensor,
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1).to(torch.int64)

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_k > 0:
        k = min(top_k, probs.size(-1))
        topk_vals, _ = torch.topk(probs, k, dim=-1)
        threshold = topk_vals[..., -1:]
        probs = probs * (probs >= threshold).float()

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum_sorted = torch.cumsum(sorted_probs, dim=-1)
        prev_cumsum = torch.cat([torch.zeros_like(cumsum_sorted[..., :1]), cumsum_sorted[..., :-1]], dim=-1)
        keep_sorted = prev_cumsum < top_p
        keep = torch.zeros_like(probs, dtype=torch.bool)
        keep.scatter_(-1, sorted_idx, keep_sorted)
        probs = probs * keep.float()

    probs = probs / probs.sum(dim=-1, keepdim=True)
    cumsum = torch.cumsum(probs, dim=-1)
    tokens = torch.sum(cumsum <= uniform_draws.unsqueeze(-1), dim=-1)

    return tokens.to(torch.int64)