# <span style="font-size: 20px;">Implement FlashAttention with Online Softmax</span>

<span style="font-size: 14px;">FlashAttention computes exact dense attention in tiles while maintaining online softmax statistics, avoiding storage of the full score and probability matrices.</span>

---

## <span style="font-size: 16px;">What FlashAttention actually changes</span>

<span style="font-size: 14px;">Dense scaled dot-product attention, and standard multi-head attention built on top of it, materialize a full $(seq_q, seq_k)$ score matrix, apply softmax over it, and then multiply by values. FlashAttention computes exactly the same mathematical result, using exactly the same formula, but restructures the computation so that only a small tile of the score matrix ever exists in memory at once.</span>

<span style="font-size: 14px;">This is a systems-level change, not an algorithmic approximation: given the same query, key, and value tensors, FlashAttention and dense attention produce identical numbers, up to ordinary floating-point rounding, for any choice of block sizes.</span>

<span style="font-size: 14px;">The motivation is purely about memory: the full score matrix, at $O(seq_q \cdot seq_k)$ entries, is frequently the largest intermediate tensor in an attention layer for long sequences, and materializing it is what limits how long a sequence can be processed at all on hardware with a fixed amount of fast memory.</span>

$$
m_i^{new} = \max(m_i, \max(S_{block}))
$$

$$
\ell_i^{new} = \ell_i \, e^{m_i - m_i^{new}} + \sum e^{S_{block} - m_i^{new}}
$$

$$
O_i^{new} = O_i \, e^{m_i - m_i^{new}} + e^{S_{block} - m_i^{new}} V_{block}
$$

---

## <span style="font-size: 16px;">Why softmax cannot simply be computed block by block</span>

<span style="font-size: 14px;">Softmax normalizes by dividing every term by the sum of all terms in its row, and the numerically stable formulation additionally subtracts the row's maximum before exponentiating.</span>

<span style="font-size: 14px;">Both the sum and the maximum are properties of the entire row, so if scores are only ever available one block at a time, neither can be computed correctly by looking at a single block in isolation: an early block might contain the largest score in the row, or it might not, and there is no way to know in advance without having already seen every block.</span>

<span style="font-size: 14px;">Online softmax solves this by tracking a running estimate of both the maximum and the sum, and correcting the running estimate every time a new block reveals a larger true maximum than was previously known.</span>

---

## <span style="font-size: 16px;">Why the running maximum requires rescaling old contributions</span>

<span style="font-size: 14px;">Suppose the first key block processed for a query row has maximum score $m_1$, and the running output and denominator are computed relative to $m_1$.</span>

<span style="font-size: 14px;">When a later key block arrives with a larger maximum $m_2 > m_1$, every term computed relative to $m_1$ is now expressed relative to the wrong reference point: $e^{x - m_1}$ needs to become $e^{x - m_2}$, and since $e^{x - m_2} = e^{x - m_1} \cdot e^{m_1 - m_2}$, this correction is achieved by multiplying every previously accumulated quantity by the single scalar factor $e^{m_1 - m_2}$.</span>

<span style="font-size: 14px;">This is the rescaling step: it retroactively corrects everything computed so far to be consistent with the newly discovered, larger maximum, without needing to revisit or recompute any individual score from an earlier block.</span>

---

## <span style="font-size: 16px;">Why this reproduces the exact softmax, not an approximation</span>

<span style="font-size: 14px;">Expanding the recurrence shows that, after processing all key blocks, the running denominator $\ell_i$ equals $\sum_k e^{S_k - m_i^{final}}$ summed over every key position, where $m_i^{final}$ is the true maximum over the entire row, exactly the numerically stable softmax denominator computed in one pass. Likewise, the running output $O_i$ equals $\sum_k e^{S_k - m_i^{final}} V_k$, the numerator of the stable softmax-weighted sum.</span>

<span style="font-size: 14px;">Dividing $O_i$ by $\ell_i$ at the end therefore produces precisely the same result as computing the entire row's stable softmax at once and multiplying by $V$; the order in which blocks are visited, and how many blocks the row is divided into, does not change this final identity, only the sequence of intermediate rescalings needed to arrive at it.</span>

---

## <span style="font-size: 16px;">Why a fully-masked-so-far state needs explicit handling</span>

<span style="font-size: 14px;">Before any key block has contributed a real score to a given row, the running maximum is $-\infty$ by convention, representing "nothing has been seen yet." With causal masking, an early query block may also encounter one or more key blocks that are entirely in the future relative to every query in that block, contributing no valid scores at all; for those rows, the running maximum can remain $-\infty$ even after a block has been processed.</span>

<span style="font-size: 14px;">Computing $e^{m_i - m_i^{new}}$ when both values are $-\infty$ is mathematically $e^{-\infty - (-\infty)}$, an indeterminate form that evaluates to NaN in floating-point arithmetic rather than the correct answer of 0; this must be special-cased explicitly, checking whether the reference maximum is $-\infty$ and substituting 0 directly, since the correct contribution of "no information yet" is unambiguously zero, not a value that happens to be computed from an undefined expression.</span>

---

## <span style="font-size: 16px;">Why causal masking must use absolute, not block-relative, positions</span>

<span style="font-size: 14px;">A query at absolute position 10 must never attend to a key at absolute position 12, regardless of which query block or key block either position happens to fall inside. If the causal mask were built using indices relative to the start of the current block, for example always masking "future" positions within a block starting from 0, the masking pattern would be correct only when a query block and key block happen to start at the same absolute offset, and silently wrong in every other case, including the common case where blocks of different sizes cause query and key block boundaries to fall out of alignment.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">Long-context inference is directly limited by how much intermediate memory an attention layer needs, and the dense score matrix, not the query, key, or value tensors themselves, is typically the dominant term for long sequences. Tiling the computation so that only one block's worth of scores exists at a time is what makes it practical to run attention over sequences far longer than would fit if the full score matrix had to be materialized, without changing the mathematical operation being computed at all.</span>

<span style="font-size: 14px;">This is why FlashAttention-style tiling is standard in virtually every production inference and training system for transformer models: it is a strict improvement in memory scaling with zero cost to correctness, in contrast to architectural changes like multi-query or grouped-query attention, which trade away some representational flexibility in exchange for their memory savings.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">The total number of floating-point operations performed across every block, summed over the whole computation, matches dense attention: $O(B \cdot seq_q \cdot seq_k \cdot d_k)$ for score computation and $O(B \cdot seq_q \cdot seq_k \cdot d_v)$ for the weighted sum, since every query-key pair is still scored exactly once regardless of how the sequence is partitioned into blocks. Tiling does not reduce the asymptotic compute cost; it changes how much of that computation's intermediate state must be resident in memory at any single point in time.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">Dense attention requires $O(B \cdot seq_q \cdot seq_k)$ memory for the score and softmax-weight tensors. Tiled online softmax requires only $O(B \cdot query\_block\_size \cdot key\_block\_size)$ for the currently active score tile, plus $O(B \cdot seq_q \cdot d_v)$ for the running per-query-block accumulators, which is independent of $seq_k$. This is the entire memory benefit of the algorithm: peak memory no longer grows with the product of the two sequence lengths, only with the chosen block sizes and the output size itself.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Computing the causal mask using block-local indices instead of the absolute query and key positions, which is correct only when query and key blocks happen to align and silently wrong otherwise.</span>

* <span style="font-size: 14px;">Omitting an explicit guard for the $-\infty - (-\infty)$ case, letting NaN enter the running accumulators and silently corrupt every later block's contribution for that row.</span>

* <span style="font-size: 14px;">Rescaling accumulated quantities using the old maximum alone, rather than the difference between the old and new maximum, which produces a plausible-looking but numerically incorrect result.</span>

* <span style="font-size: 14px;">Using a fixed-size slice for each block instead of clamping to the sequence length, which silently drops the final partial block whenever the block size does not evenly divide the sequence length.</span>

* <span style="font-size: 14px;">Believing that a smaller block size trades away numerical correctness for memory savings; correctly implemented online softmax is exact for any valid block size, including a block size of 1.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$m_i$ is the running maximum for query row $i$, $l_i$ is its running exponential denominator, $O_i$ is its running weighted-value accumulator, and each key block contributes a new score tile.</span>

<span style="font-size: 14px;">Each score tile has shape $(B,Q_b,K_b)$, row statistics have shape $(B,Q_b)$, accumulators have shape $(B,Q_b,d_v)$, and the final output has shape $(B,S_q,d_v)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: query.shape = key.shape = value.shape = (1, 4, 4), query_block_size = 4, key_block_size = 4, causal = False</span>

<span style="font-size: 14px;">Output: tensor of shape (1, 4, 4), e.g. row 0 is [0.1623, 0.0636, 0.0619, -0.3231]</span>

<span style="font-size: 14px;">Explanation: With one block covering the whole sequence, this must exactly match dense attention.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">The FlashAttention paper described an IO-aware exact attention algorithm that reduces transfers between high-bandwidth memory and on-chip storage. Its key contribution is data movement, not an approximation to softmax.</span>

<span style="font-size: 14px;">This PyTorch exercise exposes the recurrence rather than kernel-level tiling. Production CUDA or Triton kernels fuse the same logical updates so intermediate score tiles need not be written to device memory.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">The running maximum rescales both earlier and current exponentials into one common numerical frame.</span>

<span style="font-size: 14px;">The running denominator and weighted-value accumulator represent the exact softmax numerator and denominator over every key block seen so far.</span>

<span style="font-size: 14px;">Absolute query and key indices make causal masking correct across block boundaries and partial blocks.</span>

<span style="font-size: 14px;">Rows with no visible key in an early tile retain their prior state until a valid key appears, and partial edge tiles follow the same recurrence.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Compare against dense attention over several unrelated block-size pairs, including sizes that do not divide either sequence length. Large-magnitude tests specifically detect omission of the maximum-rescaling term.</span>

<span style="font-size: 14px;">Causal tests must cross tile boundaries because a block-relative triangular mask can pass single-block cases while allowing future keys in later blocks.</span>

---