import torch
from typing import Tuple

def simulate_continuous_batching(
    arrival_steps: torch.Tensor,
    required_tokens: torch.Tensor,
    max_active_sequences: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = arrival_steps.shape[0]
    
    # Track state for each request
    remaining = required_tokens.clone()
    admitted = torch.zeros(n, dtype=torch.bool)
    active = torch.zeros(n, dtype=torch.bool)
    completion_step = torch.full((n,), -1, dtype=torch.int64)

    schedule_rows = []
    step = 0
    num_completed = 0

    # Continue loop until every request finishes generating its required tokens
    while num_completed < n:
        # 1. Check if there are available slots in the current batch
        free_slots = max_active_sequences - int(active.sum().item())
        
        if free_slots > 0:
            # Find candidate requests that have arrived but haven't been admitted yet
            candidates = [
                i for i in range(n)
                if not admitted[i].item() and arrival_steps[i].item() <= step
            ]
            # Priority tie-breaker: earlier arrival step, then smallest index i
            candidates.sort(key=lambda i: (arrival_steps[i].item(), i))
            
            # Admit as many candidates as there are available slots
            for i in candidates[:free_slots]:
                admitted[i] = True
                active[i] = True

        # 2. Record which requests are active during this iteration step
        schedule_rows.append(active.clone())

        # 3. Simulate processing 1 token for each active request
        for i in range(n):
            if active[i]:
                remaining[i] -= 1
                
                # Check if the sequence finished generating all required tokens
                if remaining[i].item() == 0:
                    completion_step[i] = step
                    active[i] = False
                    num_completed += 1

        # Advance to the next time step
        step += 1

    # Combine step-by-step active masks into a 2D tensor (num_steps, num_requests)
    active_schedule = torch.stack(schedule_rows, dim=0)
    
    return active_schedule, completion_step