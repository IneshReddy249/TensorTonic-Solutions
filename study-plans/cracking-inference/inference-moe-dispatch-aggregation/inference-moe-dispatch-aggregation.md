# <span style="font-size: 20px;">Implement MoE Token Dispatch and Expert Aggregation</span>

<span style="font-size: 14px;">MoE dispatch and aggregation sends each token to its selected experts, evaluates the expert transforms, and combines routed outputs with their gate weights.</span>

---

## <span style="font-size: 16px;">From a routing decision to an actual computation</span>

<span style="font-size: 14px;">Top-k routing, the previous problem, decides which experts each token should use and how much weight each should receive, but produces no model output by itself: it is purely a decision about where computation should go next. This problem is what actually carries that decision out, running each token through its selected experts' feed-forward networks and combining the results.</span>

<span style="font-size: 14px;">The two problems are deliberately separated because they solve different kinds of problems: routing is a ranking and selection problem over logits, while dispatch and aggregation is a grouping and batched-computation problem over the tokens the routing decision selected.</span>

$$
y_t = \sum_{e \in E(t)} w_{t,e} \cdot \text{FFN}_e(x_t)
$$

$$
\text{FFN}_e(x) = \text{ReLU}(x \, W_{in}^{(e)}) \, W_{out}^{(e)}
$$

---

## <span style="font-size: 16px;">Why the expert function's exact form matters</span>

<span style="font-size: 14px;">Every expert here is a small feed-forward network, structurally identical to a standard transformer feed-forward block: an up-projection to a wider hidden dimension $d_{ff}$, a nonlinearity, and a down-projection back to $d_{model}$.</span>

<span style="font-size: 14px;">Nothing about "feed-forward expert" inherently fixes which nonlinearity is used, or whether the token vector is treated as a row multiplied on the left or a column multiplied on the right of the weight matrices; both conventions appear in real implementations, and they are not interchangeable without also transposing every weight matrix involved.</span>

<span style="font-size: 14px;">This is why the exact activation and orientation must be stated explicitly and followed precisely: a numerically different but structurally similar expert function would produce plausible-looking but wrong outputs for every single test case, not just an isolated edge case, since it changes the fundamental computation every expert performs.</span>

---

## <span style="font-size: 16px;">Why dispatch groups by expert rather than looping by token</span>

<span style="font-size: 14px;">A per-token loop that, for each token, individually looks up its selected experts and runs each one, would technically compute the correct answer, but does so by repeatedly invoking each expert's weight matrices on a single token vector at a time, one matrix-vector product after another, foregoing the ability to batch many tokens through the same expert's larger matrix multiply.</span>

<span style="font-size: 14px;">Grouping by expert instead means every expert, if selected by multiple tokens, processes all of those tokens together as one batched matrix multiply.</span>

<span style="font-size: 14px;">This is not just a performance detail confined to a single implementation choice; it is the reason production Mixture-of-Experts serving systems are built around an explicit dispatch step, gathering all tokens destined for a given expert (often across multiple devices) before that expert's computation runs at all, rather than serving each token's expert calls independently.</span>

---

## <span style="font-size: 16px;">Why empty expert groups need no special output logic</span>

<span style="font-size: 14px;">A token distribution across experts is a consequence of what the router happened to select for the tokens in this particular batch or request; there is no guarantee every expert receives at least one token, and for some Mixture-of-Experts configurations with many available experts and small batches, most experts typically receive none.</span>

<span style="font-size: 14px;">An expert selected by zero tokens contributes nothing to any token's output, which is not a special case requiring extra logic: it falls out naturally from the fact that a sum over zero selected experts contributes zero, provided the implementation does not attempt to force every expert to process a possibly-empty group in a way that could error or silently corrupt shapes.</span>

<span style="font-size: 14px;">Explicitly skipping empty groups, rather than relying on operations happening to behave correctly on zero-length inputs, keeps the intent of the computation clear and matches how a real dispatch system would skip communicating with an expert that received no tokens at all.</span>

---

## <span style="font-size: 16px;">Why the routing weight lookup needs both the token and the slot</span>

<span style="font-size: 14px;">A given token can be routed to more than one expert when top_k > 1, and its routing weight for a specific expert is not a single scalar per token; it is one weight per selected expert, corresponding to the position that expert occupies among the token's top-k choices.</span>

<span style="font-size: 14px;">When gathering "every token that selected expert $e$," it is not enough to know which tokens those are; the correct routing weight for each of them also depends on which of their top_k slots this particular expert occupies, since a token's set of routing weights was computed once, together, as a softmax over its own selected logits, and each weight in that softmax output belongs to a specific slot, not to a specific expert directly until the indices are cross-referenced.</span>

---

## <span style="font-size: 16px;">Why accumulation, not overwriting, correctly implements the aggregation sum</span>

<span style="font-size: 14px;">The aggregation formula sums a token's weighted contributions across every expert it was routed to. Because the outer loop processes one expert at a time, a token routed to two experts is visited during two separate iterations of that loop, once per expert, each time contributing one weighted term to its final output.</span>

<span style="font-size: 14px;">Using an accumulating write, rather than an overwriting one, at each contribution is what correctly reconstructs the full sum across however many iterations happen to touch a given token; a token routed to only one expert simply accumulates once, and a token routed to top_k experts accumulates top_k times, with no special-casing required for either.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">In a real serving deployment, especially one where different experts are placed on different devices, dispatch is the step that determines which tokens need to be communicated to which device before any expert computation can begin, and aggregation is the step that determines how to combine results once experts finish and their outputs return.</span>

<span style="font-size: 14px;">The correctness of routing weight lookup, per-token grouping, and accumulated aggregation covered here is exactly what a distributed all-to-all communication and combine step must replicate at much larger scale; understanding it correctly on a single device, with explicit tensors rather than cross-device messages, is the foundation for reasoning about expert-parallel serving, covered later in this study plan, where the same logical operation is spread across multiple GPUs.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Routing costs $O(N \cdot E \log E)$, as established previously. Across the entire expert loop, every one of the $N \cdot top\_k$ total (token, expert) selections is processed by exactly one expert's feed-forward computation, so the total feed-forward cost is $O(N \cdot top\_k \cdot d_{model} \cdot d_{ff})$, independent of how unevenly tokens happen to be distributed among experts, since the sum of tokens processed across all experts is always exactly $N \cdot top\_k$.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">Each expert's iteration only ever holds tensors sized by however many tokens it actually received, not by the total token count or total expert count; an expert that receives no tokens uses essentially no memory for that iteration. The final output tensor is $O(N \cdot d_{model})$, independent of top_k or num_experts, since aggregation reduces every token's contributions down to a single output row regardless of how many experts contributed to it.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Running every expert on every token and masking the result afterward, rather than gathering only the tokens routed to each expert, which is correct but discards the sparsity that makes Mixture-of-Experts computationally efficient in the first place.</span>

* <span style="font-size: 14px;">Using the wrong activation function or an implicitly transposed weight orientation for the expert feed-forward computation, producing plausible but systematically wrong outputs across every test case.</span>

* <span style="font-size: 14px;">Losing track of which top_k slot a selected expert corresponds to for a given token, and looking up the wrong routing weight as a result.</span>

* <span style="font-size: 14px;">Overwriting a token's output on each contributing expert's iteration instead of accumulating, which silently keeps only the last expert's contribution instead of the correct weighted sum.</span>

* <span style="font-size: 14px;">Assuming every expert receives at least one token and writing code that breaks, rather than simply doing no work, when an expert's group is empty.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$X_t$ is token $t$, $I_{t,r}$ and $G_{t,r}$ identify its expert and gate at route slot $r$, $E_e(cdot)$ is expert $e$, and $Y_t=sum_r G_{t,r}E_{I_{t,r}}(X_t)$.</span>

<span style="font-size: 14px;">Token states and outputs have shape $(T,D)$; router logits have shape $(T,E)$; expert weights carry an expert axis; routed items have logical shape $(T,k,D)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: token_states.shape = (2, 3), num_experts = 1, top_k = 1</span>

<span style="font-size: 14px;">Output: tensor([[-0.9935, 0.2688, 1.1763], [-0.6845, 0.8316, 1.3869]])</span>

<span style="font-size: 14px;">Explanation: With only one expert, every token routes to it with weight 1.0.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">Routing alone does not realize sparse computation. An inference engine must convert route choices into expert-local batches, execute those batches efficiently, and restore token order for the next dense layer.</span>

<span style="font-size: 14px;">Single-device grouping models the same semantics used by expert-parallel all-to-all systems. Physical communication changes placement and cost, but the weighted token-level sum remains the correctness contract.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">Grouping routed token copies by expert changes execution order but not which expert processes each copy.</span>

<span style="font-size: 14px;">The route-slot lookup keeps every expert output paired with the gate that selected it.</span>

<span style="font-size: 14px;">Accumulation by original token index implements the weighted sum even when a token has multiple routes.</span>

<span style="font-size: 14px;">An expert may receive no routed items, and repeated token indices must accumulate rather than overwrite earlier contributions.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">A direct token-by-token reference is useful because it is simple and independent of grouped dispatch. Tests should include empty experts, top-k one, top-k greater than one, and different expert selections for adjacent tokens.</span>

<span style="font-size: 14px;">Conservation of route associations matters more than group order: reordering dispatched work is valid only if inverse mapping and routing weights reconstruct outputs in original token order.</span>

---