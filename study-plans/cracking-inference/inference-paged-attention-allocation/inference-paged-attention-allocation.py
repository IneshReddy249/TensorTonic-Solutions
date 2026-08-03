import torch
from typing import List, Tuple

def allocate_kv_blocks(
    seq_lengths: List[int],
    block_size: int,
    free_block_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (block_table, blocks_used, remaining_free_blocks)
    """
    blocks_needed = []

    for length in seq_lengths:
        n = (length + block_size - 1) // block_size
        blocks_needed.append(n)

    total_needed = sum(blocks_needed)
    total_available = len(free_block_ids)

    if total_needed > total_available:
        raise  RuntimeError("Not enough free blocks!")

    max_blocks = max(blocks_needed)

    block_table = []
    free_idx = 0

    for n in blocks_needed:
        assigned_blocks = free_block_ids[free_idx : free_idx + n]
        free_idx += n

        padding_row = assigned_blocks + [-1] * (max_blocks - len(assigned_blocks))
        block_table.append(padding_row)

    block_table_tensor = torch.tensor(block_table, dtype=torch.int64)
    blocks_used_tensor = torch.tensor(blocks_needed, dtype = torch.int64)
    remaining_free_blocks_tensor = torch.tensor(free_block_ids[free_idx:], dtype=torch.int64)

    return block_table_tensor, blocks_used_tensor, remaining_free_blocks_tensor

    pass
