import torch
from typing import List, Tuple

def match_prefix_cache(
    request_token_ids: List[int],
    cached_token_blocks: List[List[List[int]]],
    cached_physical_block_ids: List[List[int]],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (matched_token_count scalar tensor, reusable_physical_block_ids tensor)
    """
    # Step 1: Split the incoming request into complete chunks of size `block_size`
    num_completed_blocks = len(request_token_ids) // block_size
    request_blocks = []

    for i in range(num_completed_blocks):   # i will be 0, then 1, then 2
        start = i * block_size          # start index
        end   = (i + 1) * block_size    # end index
        block = request_token_ids[start:end]
        request_blocks.append(block)

    best_match_len = 0
    best_physical_ids = []
    best_candidate_idx = float("-inf")

    for cand_idx, candidate in enumerate(cached_token_blocks):
        matched_blocks = 0

        #compare blocks from the scratch
        for req_block, cand_block in zip(request_blocks, candidate):
            if req_block == cand_block:
                matched_blocks += 1
            else:
                break

        if matched_blocks > best_match_len:
            best_match_len = matched_blocks
            best_physical_ids = cached_physical_block_ids[cand_idx][:matched_blocks]

    matched_token_count = torch.tensor(best_match_len * block_size, dtype=torch.int64)
    reusable_physical_block_ids = torch.tensor(best_physical_ids, dtype = torch.int64)

    return matched_token_count, reusable_physical_block_ids
    
    pass
