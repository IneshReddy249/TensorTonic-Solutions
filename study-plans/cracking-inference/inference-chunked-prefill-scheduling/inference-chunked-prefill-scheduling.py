import torch


def schedule_chunked_prefill(
    arrival_steps: torch.Tensor,
    prompt_tokens: torch.Tensor,
    decode_tokens: torch.Tensor,
    max_tokens_per_iteration: int,
    prefill_chunk_size: int,
) -> torch.Tensor:
    """
    Returns: (num_events, 4) int64 tensor with columns
    [iteration, request_id, phase(0=prefill,1=decode), tokens_processed]
    """
    n = len(arrival_steps)

    arrival = [int(x) for x in arrival_steps]
    rem_prefill = [int(x) for x in prompt_tokens]
    rem_decode = [int(x) for x in decode_tokens]

    # iteration at which each request FINISHED prefill; None = not finished yet
    prefill_done_at = [None] * n

    # FCFS order: earlier arrival first, original index breaks ties
    order = sorted(range(n), key=lambda i: (arrival[i], i))

    events = []
    t = 0

    while any(d > 0 for d in rem_decode):
        budget = max_tokens_per_iteration
        decode_rows = []
        prefill_rows = []

        # ---- PHASE 1: decode (strict priority) ----
        for i in range(n):
            if budget <= 0:
                break
            if prefill_done_at[i] is None:      # still prefilling
                continue
            if prefill_done_at[i] >= t:         # finished THIS iteration -> wait
                continue
            if rem_decode[i] <= 0:              # already fully decoded
                continue
            decode_rows.append([t, i, 1, 1])
            rem_decode[i] -= 1
            budget -= 1

        # ---- PHASE 2: prefill (leftover budget only, arrival order) ----
        for i in order:
            if budget <= 0:
                break
            if arrival[i] > t:                  # hasn't arrived yet
                continue
            if rem_prefill[i] <= 0:             # nothing left to prefill
                continue
            work = min(budget, prefill_chunk_size, rem_prefill[i])
            prefill_rows.append([t, i, 0, work])
            rem_prefill[i] -= work
            budget -= work
            if rem_prefill[i] == 0:
                prefill_done_at[i] = t

        # decode rows first (sorted by id), then prefill rows in arrival order
        decode_rows.sort(key=lambda r: r[1])
        events.extend(decode_rows)
        events.extend(prefill_rows)

        t += 1

    return torch.tensor(events, dtype=torch.int64).reshape(-1, 4)