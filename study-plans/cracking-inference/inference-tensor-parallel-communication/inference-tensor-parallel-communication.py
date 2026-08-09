import torch

def tensor_parallel_allreduce_cost(
    message_element_count: int,
    bytes_per_element: int,
    tensor_parallel_world_size: int,
    link_bandwidth: float,
    per_hop_latency: float,
    collective_count: int,
) -> torch.Tensor:
    """
    Returns: torch.float64 tensor of shape (2,): [total_communicated_bytes, total_communication_time]
    Both values are per rank, aggregated across collective_count repeated all-reduces.
    """

    n= tensor_parallel_world_size
    
    # 1. Calculate the total size of the message in bytes
    message_bytes = message_element_count * bytes_per_element

    # 2. Calculate the number of steps in the ring
    ring_steps = 2 * (n - 1)

    # 3. Calculate the bytes transmitted by a single rank per collective
    per_rank_bytes = (ring_steps * message_bytes) / n

    # 4. Calculate the time taken for a single collective
    time = (ring_steps * per_hop_latency) + (per_rank_bytes / link_bandwidth)
 
    
    # 5. Scale by the number of collectives
    total_bytes =  per_rank_bytes * collective_count
    total_time = time * collective_count

    return torch.tensor([total_bytes, total_time], dtype=torch.float64)
    pass
