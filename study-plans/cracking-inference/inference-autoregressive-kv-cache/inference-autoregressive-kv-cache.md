# <span style="font-size: 20px;">Implement Autoregressive Decoding with a KV Cache</span>

<span style="font-size: 14px;">Autoregressive KV caching reuses keys and values from earlier positions so each decoding step computes attention only for the new query.</span>

---

## <span style="font-size: 16px;">Why decoding is fundamentally different from a single forward pass</span>

<span style="font-size: 14px;">Every attention problem covered so far in this study plan takes a complete sequence and produces attention outputs for every position in one computation. Autoregressive generation does not work that way: a language model produces one token at a time, and each new token's generation requires attention over every position generated so far, including the one just produced.</span>

<span style="font-size: 14px;">Naively handling this by re-running the full forward pass over the entire sequence-so-far after every new token would work correctly, but would redundantly recompute every earlier position's key and value projections at every single step, work whose result cannot have changed since those earlier positions' inputs have not changed. The key/value cache exists to eliminate exactly this redundant recomputation.</span>

$$
\text{cache}_t = \text{cache}_{t-1} \Vert (k_t, v_t)
$$

$$
o_t = \text{softmax}\!\left(\frac{q_t \, \text{cache}_t^K{}^\top}{\sqrt{d_k}}\right) \text{cache}_t^V
$$

---

## <span style="font-size: 16px;">Why the cache only needs to grow, never shrink or rewrite</span>

<span style="font-size: 14px;">Once a position's key and value are computed, they never change: they depend only on that position's own input, not on anything that happens afterward in the sequence.</span>

<span style="font-size: 14px;">This is precisely what makes a pure append-only cache correct: there is no scenario in causal, left-to-right generation where an already-cached key or value needs to be revised in light of later tokens, because attention itself, by the causal masking convention already established for standard attention, never lets a later token influence an earlier one's representation in the first place.</span>

<span style="font-size: 14px;">If revision were ever needed, an append-only cache would be insufficient; the fact that it is sufficient here is a direct consequence of causality, not an incidental implementation convenience.</span>

---

## <span style="font-size: 16px;">Why processing one position at a time is equivalent to a mask, without needing one</span>

<span style="font-size: 14px;">Dense causal attention enforces "position $t$ cannot attend to position $k > t$" with an explicit mask applied to the full score matrix. The incremental formulation achieves the identical restriction implicitly: at the moment position $t$'s attention is computed, the cache physically does not yet contain any position beyond $t$, since those positions have not been processed yet. There is nothing to mask, because there is nothing later to accidentally attend to.</span>

<span style="font-size: 14px;">This is a genuinely different way of enforcing the same constraint, not merely a notational convenience: it reflects how a real generation loop actually works, where future tokens do not exist yet at the time a given token's output is being computed, in contrast to a training-time forward pass, where the entire sequence already exists and a mask is needed to hide the parts a position should not see.</span>

---

## <span style="font-size: 16px;">Why this must produce identical results to dense causal attention, not merely similar ones</span>

<span style="font-size: 14px;">For any position $t$, the incremental algorithm computes attention using precisely the sub-sequence key[:t+1] and value[:t+1], since that is exactly what has been appended to the cache by the time position $t$'s query is processed. This is also exactly the set of keys and values that a causal mask would leave visible to position $t$ in a dense, whole-sequence computation.</span>

<span style="font-size: 14px;">Since the underlying scaled dot-product attention formula, scaling, and softmax are identical in both formulations, and they are applied to the identical subset of keys and values at every position, the two approaches are mathematically the same computation carried out in a different order, not an approximation of each other; any implementation that produces even slightly different numbers between the two has a bug in one of them, not an inherent limitation of caching.</span>

---

## <span style="font-size: 16px;">Why the returned cache must equal the full input tensors exactly</span>

<span style="font-size: 14px;">Testing that the incremental cache-building process is correct cannot rely only on the final attention outputs matching dense attention, since a bug that corrupts the cache's structure without breaking every single position's masking could still coincidentally produce correct outputs for parts of the sequence, especially the last few positions where the correct set of visible keys happens to be large.</span>

<span style="font-size: 14px;">Independently checking that the returned key_cache and value_cache exactly equal the original, complete key and value tensors verifies the append-only construction directly, position by position, rather than only observing its downstream numerical effect on attention outputs.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">This is the foundational mechanism behind essentially every optimization covered in the remainder of this study plan's decoding and serving sections: grouped-query and multi-query attention exist specifically to reduce how much this cache costs to store per token; multi-head latent attention compresses what gets cached; PagedAttention, covered next, changes how cache memory for many concurrent requests is physically allocated; prefix caching reuses cache entries across requests that happen to share a prefix; and speculative decoding still ultimately needs a correct, appendable cache to verify draft tokens against.</span>

<span style="font-size: 14px;">Every one of those techniques assumes, as a starting point, that key/value caching itself works exactly as implemented here: append once per generated token, never recompute, never revise.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Processing position $t$ costs $O(B \cdot (t+1) \cdot d_k)$ for scores and a comparable amount for the weighted sum, since the cache holds $t+1$ entries. Summed across all $seq$ positions, the total cost is $O(B \cdot seq^2 \cdot d_k)$, identical to the total cost of dense causal attention computed all at once; incremental caching redistributes when this cost is paid, spreading it across many small steps as generation proceeds, rather than reducing the total amount of computation required.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">At any point during generation, the cache holds exactly as many entries as have been generated so far, no more and no less: $O(B \cdot t \cdot d_k)$ at step $t$, growing to $O(B \cdot seq \cdot d_k)$ only once the full sequence has been produced. This incremental growth is what makes it possible to begin generating without knowing the eventual sequence length in advance, and is the property that every later cache-memory optimization in this study plan, compression, sharing, paging, and eviction, builds directly on top of.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Recomputing attention or projections for earlier positions at every step instead of only appending the current position, which produces correct results but defeats the entire purpose of caching.</span>

* <span style="font-size: 14px;">Allowing a later position's query to see cache entries appended after it, which would require passing a different, larger cache slice than what has legitimately been generated at that point.</span>

* <span style="font-size: 14px;">Rebuilding the cache from scratch at each step instead of concatenating onto the previous step's cache, which is both wasteful and risks subtly duplicating or dropping entries if the rebuild logic has an off-by-one error.</span>

* <span style="font-size: 14px;">Returning attention outputs that happen to be numerically close to dense causal attention's outputs without independently verifying that the cache itself contains exactly the right entries in the right order.</span>

* <span style="font-size: 14px;">Assuming the value dimension must equal the key dimension, and writing shape logic that silently breaks when $d_v \neq d_k$.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$q_t$, $k_t$, and $v_t$ are the query, key, and value at decode step $t$; $K_{0:t}$ and $V_{0:t}$ are the caches after appending that step; the output is $o_t=softmax(q_tK_{0:t}^{T}/sqrt{d})V_{0:t}$.</span>

<span style="font-size: 14px;">Inputs, outputs, and final caches have shape $(B,S,D)$; at step $t$, each cache has shape $(B,t+1,D)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: query.shape = key.shape = value.shape = (1, 1, 4)</span>

<span style="font-size: 14px;">Output: outputs[0, 0] = [0.8, 0.17, 0.09, -0.61]; key_cache and value_cache equal the input key and value exactly</span>

<span style="font-size: 14px;">Explanation: With one position, attention has only itself to attend to.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">Transformer language models generate one token at a time, but keys and values for an existing prefix do not change. A KV cache trades persistent memory for avoiding repeated projection work on that prefix.</span>

<span style="font-size: 14px;">The cache also becomes serving state. Its size affects maximum context length, batch capacity, paging policy, prefix reuse, and the cost of moving a request between workers.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">Appending one key and value at each step makes the cache contain exactly the visible causal prefix.</span>

<span style="font-size: 14px;">Attending the current query only to that prefix is equivalent to applying a triangular mask in dense causal attention.</span>

<span style="font-size: 14px;">Concatenating step outputs restores the original sequence order while leaving the final caches unchanged.</span>

<span style="font-size: 14px;">The first token attends only to itself, which provides a simple exact check of both output and cache initialization.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Compare every step output with the corresponding row of one dense causal-attention call and compare both final caches directly with the input key and value tensors.</span>

<span style="font-size: 14px;">The first row, last row, multi-token sequence, and multi-batch cases detect off-by-one appends and accidental cache sharing between batch elements.</span>

---

## <span style="font-size: 16px;">Behavioral contract</span>

<span style="font-size: 14px;">Accept query, key, and value sequences with matching batch, sequence, and feature dimensions.</span>

<span style="font-size: 14px;">Return the causal attention output for every input position plus the final key and value caches.</span>

<span style="font-size: 14px;">The final caches must equal the complete input key and value sequences in their original order.</span>

<span style="font-size: 14px;">Output position $t$ must depend only on keys and values from positions zero through $t$.</span>

<span style="font-size: 14px;">The returned outputs must match dense causal attention within floating-point tolerance.</span>

<span style="font-size: 14px;">Support batch sizes and sequence lengths greater than one.</span>

<span style="font-size: 14px;">All returned tensors must be finite and preserve the input dtype.</span>

---

## <span style="font-size: 16px;">Operational cost</span>

<span style="font-size: 14px;">This direct educational simulation performs $O(BS^2D)$ attention arithmetic across all steps.</span>

<span style="font-size: 14px;">Persistent cache memory is $O(BSD)$ for keys and values, while each step avoids recomputing prior projections.</span>

---