import torch
from typing import Tuple

def compute_serving_metrics(
    arrival_times: torch.Tensor,
    first_token_times: torch.Tensor,
    token_timestamps: torch.Tensor,
    output_token_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # Step 0 — cast
    arrival = arrival_times.float()
    first   = first_token_times.float()
    ts      = token_timestamps.float()
    n       = output_token_counts.long()

    # Step 1 — TTFT
    ttft = first - arrival

    # Step 2 — last REAL token per request
    last_idx = (n - 1).clamp(min=0).unsqueeze(1)
    last     = ts.gather(1, last_idx).squeeze(1)

    # Step 3 — TPOT
    denom = (n - 1).float().clamp(min=1.0)
    tpot  = torch.where(n > 1, (last - first) / denom, torch.zeros_like(first))

    # Step 4 — mean ITL
    mean_itl = tpot.clone()

    # Step 5 — throughput
    total_tokens = n.sum().float()
    span         = last.max() - arrival.min()
    throughput   = total_tokens / span

    return ttft, tpot, mean_itl, throughput