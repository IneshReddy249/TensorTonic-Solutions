import torch

def estimate_inference_gpu_count(
    parameter_count: int,
    bytes_per_parameter: float,
    kv_cache_bytes: int,
    activation_bytes: int,
    runtime_overhead_fraction: float,
    usable_bytes_per_gpu: int,
) -> torch.Tensor:
    """
    Returns: torch.int64 tensor of shape (2,): [total_required_bytes, min_gpu_count]
    """

    weight_bytes = parameter_count * bytes_per_parameter
    base_bytes = weight_bytes + kv_cache_bytes + activation_bytes
    total_with_overhead = base_bytes * (1.0 + runtime_overhead_fraction)
    total_required_bytes = math.ceil(total_with_overhead)

    min_gpu_count = math.ceil(total_required_bytes / usable_bytes_per_gpu)

    return torch.tensor([total_required_bytes, min_gpu_count], dtype=torch.int64)
    
    pass
