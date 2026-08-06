import torch
from typing import Tuple

def schedule_dynamic_batches(
    arrival_times: torch.Tensor,
    max_batch_size: int,
    max_queue_delay: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(arrival_times)

    # 1. sort requests by (arrival time, original index)
    order = sorted(range(n), key=lambda i: (arrival_times[i].item(), i))

    assignment = [0] * n      # which batch each request landed in
    start_time = [0.0] * n    # when that batch left

    batch_id = -1             # counter, first real batch will be 0
    members = []              # original indices sitting in the current batch
    deadline = None           # when the current batch MUST leave

    def close_batch(dispatch_time):
        for idx in members:
            assignment[idx] = batch_id
            start_time[idx] = dispatch_time

    # 2. walk requests in sorted order
    for i in order:
        a = arrival_times[i].item()

        # 2a. did the open batch already time out before this person arrived?
        if deadline is not None and a >= deadline:
            close_batch(deadline)
            members = []
            deadline = None

        # 2b. no batch open? start a fresh one with this request
        if not members:
            batch_id += 1
            deadline = a + max_queue_delay

        # 2c. put the request in the current batch
        members.append(i)

        # 2d. batch full? send it right now, at this arrival time
        if len(members) == max_batch_size:
            close_batch(a)
            members = []
            deadline = None

    # 3. leftover batch leaves at its deadline
    if members:
        close_batch(deadline)

    return torch.tensor(assignment), torch.tensor(start_time)