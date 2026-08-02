# <span style="font-size: 20px;">Calculate KV Cache Memory for MHA, MQA, GQA, and MLA</span>

<span style="font-size: 14px;">KV-cache accounting converts attention architecture choices into exact memory requirements for MHA, MQA, GQA, and a scoped MLA cache.</span>

---

## <span style="font-size: 16px;">From an architectural idea to a concrete memory number</span>

<span style="font-size: 14px;">The preceding five problems each established a different attention architecture and, along the way, an informal claim about how much KV-cache memory it needs relative to standard multi-head attention: MQA is described as sharing one KV head across all query heads, GQA as sharing groups, MLA as caching a compressed latent.</span>

<span style="font-size: 14px;">This problem turns those qualitative claims into an exact, checkable formula, which matters because "smaller cache" is not useful information on its own when deciding whether a given model and context length actually fit in a given amount of GPU memory; the actual byte count is what a serving system needs.</span>

$$
\text{bytes}_{MHA/MQA/GQA} = batch \cdot seq \cdot layers \cdot 2 \cdot n_{kv} \cdot d_{head} \cdot elem\_bytes
$$

$$
\text{bytes}_{MLA} = batch \cdot seq \cdot layers \cdot (d_{latent} + d_{rope}) \cdot elem\_bytes
$$

---

## <span style="font-size: 16px;">Why the formula has exactly this shape for MHA, MQA, and GQA</span>

<span style="font-size: 14px;">Every cached token needs a key vector and a value vector, one pair per KV head, per layer, per sequence in the batch; this is why the formula multiplies batch_size, seq_len, num_layers, and 2 (for the key and the value tensor) together as a common factor across all three variants.</span>

<span style="font-size: 14px;">The only quantity that differs between MHA, MQA, and GQA is how many independent KV heads exist per layer: MHA gives every query head its own, so its KV-head count equals num_query_heads; MQA collapses every query head onto a single shared KV head, so its count is fixed at 1; GQA sits at whatever intermediate count the caller specifies.</span>

<span style="font-size: 14px;">Because this is the only difference, the three variants are not three unrelated formulas that happen to look similar; they are the same formula with one parameter substituted, which is exactly why the boundary equivalences (GQA at gqa_kv_heads = num_query_heads matching MHA, and at gqa_kv_heads = 1 matching MQA) hold as an algebraic identity, not a coincidence that needs separate verification logic.</span>

---

## <span style="font-size: 16px;">Why MLA's formula has a genuinely different shape, not just different numbers</span>

<span style="font-size: 14px;">MLA does not store separate per-KV-head key and value tensors at all; as established in the multi-head latent attention problem, it stores one shared compressed latent per token, from which full-width keys and values for every head are reconstructed on demand.</span>

<span style="font-size: 14px;">This means MLA's cache formula cannot be expressed as "the MHA/MQA/GQA formula with some particular KV-head count substituted in": there is no head-count multiplier at all, and there is no factor of 2 for separate key and value tensors, because there is only one cached object per token, not two. The formula instead multiplies the shared batch, sequence, and layer factors by the latent's own width, which is a structurally different quantity from a per-head dimension.</span>

---

## <span style="font-size: 16px;">Why the rotary-key component must be accounted for explicitly</span>

<span style="font-size: 14px;">Rotary position embeddings rotate a key vector by an angle that depends on its absolute position, and this rotation does not commute cleanly with an arbitrary low-rank compression: compressing a rotated key through a shared linear latent can lose the positional information RoPE relies on, unless positional information is handled separately from the part of the key that gets compressed.</span>

<span style="font-size: 14px;">Practical MLA implementations address this by decoupling a small, separately cached rotary-key component from the larger compressible latent, so that positional information survives exactly while the bulk of the key's content still benefits from compression.</span>

<span style="font-size: 14px;">Because this is an implementation convention rather than a universal mathematical necessity, and because different sources describe MLA's caching behavior with varying levels of precision about this detail, this problem fixes the convention explicitly: the cached MLA state per token is mla_latent_dim + mla_rotary_key_dim, two separate widths added together, rather than assuming the rotary component either does not exist or is already folded into mla_latent_dim.</span>

---

## <span style="font-size: 16px;">Why realistic inputs can overflow a 32-bit integer</span>

<span style="font-size: 14px;">A frontier-scale model with 80 layers, 64 attention heads, a head dimension of 128, FP16 weights, a batch of 32 concurrent sequences, and a context length of 128K tokens produces a KV-cache byte count in the tens of terabytes when computed for standard multi-head attention, a number that exceeds $2^{31} - 1$, the maximum value representable in a signed 32-bit integer, by a wide margin.</span>

<span style="font-size: 14px;">This is not a contrived edge case; it is the realistic regime this formula exists to be used in, which is exactly why the problem requires 64-bit integer arithmetic throughout and a torch.int64 result: a calculation that silently overflows and wraps around would report a memory estimate that is not merely imprecise but qualitatively wrong, potentially by orders of magnitude, which is far more dangerous for a capacity-planning calculation than an estimate that is merely rounded.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">This calculation is exactly the arithmetic a serving system performs, explicitly or implicitly, when deciding how many concurrent requests it can admit at a given context length, or how much context length it can support at a given batch size, before running out of GPU memory.</span>

<span style="font-size: 14px;">The relative comparison between architectures is precisely what motivates choosing GQA or MLA over standard multi-head attention for a production model in the first place: this problem makes that motivation concrete and checkable, rather than a qualitative claim about "smaller cache," and later problems in this study plan that address prefix caching, disaggregated serving, and throughput-under-an-SLO all build on being able to compute exactly this kind of memory budget correctly.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">The computation is $O(1)$: a fixed, small number of multiplications regardless of how large any individual input value is. There is no loop, no iteration over tokens, layers, or heads; the entire cache size for an entire sequence and an entire model is captured by closed-form arithmetic.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">Computing this estimate requires only $O(1)$ working memory beyond the four returned values; nothing proportional to the actual cache size being estimated is ever allocated, which is precisely the point of estimating memory analytically rather than by actually constructing the tensors involved.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Forgetting the factor of 2 for separate key and value tensors in the MHA, MQA, or GQA formulas, undercounting by half.</span>

* <span style="font-size: 14px;">Applying MLA's formula as though it needed the same factor of 2 or a head-count multiplier, when it structurally has neither.</span>

* <span style="font-size: 14px;">Omitting mla_rotary_key_dim entirely, undercounting MLA's true cache footprint by whatever fraction that component represents.</span>

* <span style="font-size: 14px;">Performing intermediate arithmetic in a fixed-width 32-bit type before the final result is assembled, silently overflowing for realistic large-scale configurations even if the final tensor dtype is correctly specified as 64-bit.</span>

* <span style="font-size: 14px;">Assuming gqa_kv_heads and num_query_heads boundary behavior needs a separate conditional branch, rather than recognizing it as the same formula naturally producing equal results at those parameter values.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$B$ is batch size, $S$ is cached sequence length, $L$ is layer count, $H_q$ is query-head count, $H_{kv}$ is KV-head count, $D$ is head width, $D_c$ is MLA latent width, and $b$ is bytes per element.</span>

<span style="font-size: 14px;">The function returns a length-four tensor ordered as MHA, MQA, GQA, and MLA memory in bytes.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: batch_size=1, seq_len=128, num_layers=1, num_query_heads=8, gqa_kv_heads=2, head_dim=64, mla_latent_dim=32, mla_rotary_key_dim=16, bytes_per_element=2</span>

<span style="font-size: 14px;">Output: tensor([262144, 32768, 65536, 12288])</span>

<span style="font-size: 14px;">Explanation: MHA: 1*128*1*2*8*64*2 = 262144. MQA: 1*128*1*2*1*64*2 = 32768. GQA: 1*128*1*2*2*64*2 = 65536. MLA: 1*128*1*(32+16)*2 = 12288.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">KV-cache sizing often determines whether a serving configuration is feasible before kernel speed is considered. Cache bytes grow linearly with batch, context length, and layer count, so architectural head sharing compounds across all three factors.</span>

<span style="font-size: 14px;">The MHA, MQA, and GQA boundary equalities are useful for both model design and capacity planning. MLA uses a different stored representation, so its width must be counted from the actual cache contract rather than inferred from query heads.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">Every cache formula multiplies independent batch, token, layer, stored-width, and element-size factors exactly once.</span>

<span style="font-size: 14px;">The factor of two applies to architectures that store separate keys and values, while MLA follows its stated latent-plus-rotary contract.</span>

<span style="font-size: 14px;">Using 64-bit integer arithmetic prevents silent overflow for realistic deployments.</span>

<span style="font-size: 14px;">When GQA uses all query heads it equals MHA; when it uses one KV head it equals MQA.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Use small hand-computable cases for dimensional reasoning, then include a large case above $2^{31}-1$ bytes to verify integer width. Check both GQA boundary equalities exactly.</span>

<span style="font-size: 14px;">Units should be tracked explicitly: element counts become bytes only after multiplying by bytes per element, and no binary or decimal size conversion belongs inside this function.</span>

---

## <span style="font-size: 16px;">Behavioral contract</span>

<span style="font-size: 14px;">Accept positive model, batch, sequence, head, latent, and element-size parameters.</span>

<span style="font-size: 14px;">Return one 64-bit integer memory count for each of MHA, MQA, GQA, and MLA.</span>

<span style="font-size: 14px;">MHA memory must include one key and one value for every query head, token, layer, and batch item.</span>

<span style="font-size: 14px;">MQA memory must include exactly one key head and one value head per token and layer.</span>

<span style="font-size: 14px;">GQA memory must use the supplied KV-head count rather than the query-head count.</span>

<span style="font-size: 14px;">MLA memory must include the compressed latent and the separate rotary-key component defined by the problem.</span>

<span style="font-size: 14px;">Results must remain exact when byte counts exceed signed 32-bit range.</span>

---