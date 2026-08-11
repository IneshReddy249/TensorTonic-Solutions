import torch
from typing import Tuple


def simulate_disaggregated_serving(
    arrival_times: torch.Tensor,
    prompt_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    prefill_replica_rates: torch.Tensor,
    decode_replica_rates: torch.Tensor,
    kv_transfer_bandwidth: float,
    fixed_transfer_latency: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = arrival_times.numel()

    arrivals = [float(x) for x in arrival_times.tolist()]
    prompts  = [float(x) for x in prompt_lengths.tolist()]
    outputs  = [float(x) for x in output_lengths.tolist()]
    p_rates  = [float(x) for x in prefill_replica_rates.tolist()]
    d_rates  = [float(x) for x in decode_replica_rates.tolist()]
    bw  = float(kv_transfer_bandwidth)
    lat = float(fixed_transfer_latency)

    p_avail = [0.0] * len(p_rates)
    d_avail = [0.0] * len(d_rates)

    prefill_idx = [-1] * n
    decode_idx  = [-1] * n
    kv_ready = [0.0] * n
    ttft     = [0.0] * n
    comp     = [0.0] * n

    # ---- Stage 1: prefill, in arrival order ----
    for i in sorted(range(n), key=lambda i: (arrivals[i], i)):
        p = min(range(len(p_avail)), key=lambda r: (p_avail[r], r))
        start = max(arrivals[i], p_avail[p])
        end = start + prompts[i] / p_rates[p]
        p_avail[p] = end
        prefill_idx[i] = p
        kv_ready[i] = end + prompts[i] / bw + lat

    # ---- Stage 2: decode, in kv_ready order ----
    for i in sorted(range(n), key=lambda i: (kv_ready[i], i)):
        d = min(range(len(d_avail)), key=lambda r: (d_avail[r], r))
        start = max(kv_ready[i], d_avail[d])
        ttft[i] = start + 1.0 / d_rates[d]
        comp[i] = start + outputs[i] / d_rates[d]
        d_avail[d] = comp[i]
        decode_idx[i] = d

    assigned = torch.tensor(list(zip(prefill_idx, decode_idx)), dtype=torch.long)
    return assigned, torch.tensor(ttft, dtype=torch.float64), torch.tensor(comp, dtype=torch.float64)