# <span style="font-size: 20px;">Implement Sparse MoE Top-k Expert Routing</span>

<span style="font-size: 14px;">Sparse mixture-of-experts routing selects a small expert set for each token and assigns normalized weights only across those selected experts.</span>

---

## <span style="font-size: 16px;">Why Mixture-of-Experts needs a routing decision at all</span>

<span style="font-size: 14px;">A dense feed-forward layer applies the same weights to every token. A Mixture-of-Experts layer instead maintains several independent feed-forward networks, called experts, and processes each token through only a small subset of them, chosen per token by a lightweight router.</span>

<span style="font-size: 14px;">This is a deliberate tradeoff: the model's total parameter count can grow very large, since most experts are inactive for any given token, while the actual compute cost per token stays close to that of a single expert, since only the selected experts are ever evaluated for that token. Everything covered in this problem, ranking, selection, and weighting, is the mechanism that decides which experts a token actually uses and how much each contributes.</span>

$$
E(t) = \text{top-}k\text{ indices of } z_t
$$

$$
w_t = \text{softmax}(z_t[E(t)])
$$

---

## <span style="font-size: 16px;">Why routing is top-k, not a hard single choice or a full soft mixture</span>

<span style="font-size: 14px;">Routing every token to a single expert (top_k=1) is the sparsest possible choice and gives the largest compute savings, but forces a single expert to represent everything about that token with no ability to blend information from multiple specializations. Routing every token through every expert with soft weights is the opposite extreme: full representational flexibility, but no sparsity benefit at all, since every expert must still be evaluated for every token.</span>

<span style="font-size: 14px;">Top-k routing sits between these: a token is routed to a handful of experts, whichever ones the router currently considers most relevant, and its output is a weighted combination of just those, capturing some of soft-mixture's blending ability while still skipping the large majority of experts per token.</span>

---

## <span style="font-size: 16px;">Why ranking must happen before any weighting</span>

<span style="font-size: 14px;">The router's job, deciding which experts matter for a token, is fundamentally a ranking problem: "which experts have the highest logits" only makes sense once every expert's logit for that token has been compared against every other. This is why the whole row of router_logits is sorted first, rather than attempting to identify the top-k experts and their weights in a single pass; the correct top-k set cannot be determined incrementally without effectively performing a sort or an equivalent selection procedure.</span>

---

## <span style="font-size: 16px;">Why softmax is computed over the selected experts, not the full expert set</span>

<span style="font-size: 14px;">Two superficially similar procedures are not equivalent: computing softmax over all num_experts logits and then keeping only the top-k resulting probabilities, versus computing softmax directly over just the top-k logits. The first approach's probabilities are influenced by every excluded expert's logit through the softmax normalization constant, so a very large logit among the excluded experts would depress the weights assigned to the selected experts even though that expert is never used.</span>

<span style="font-size: 14px;">The second approach treats the selected experts as the entire universe of consideration once routing has been decided, which is the standard convention in production sparse MoE architectures: routing weights describe how to combine the chosen experts' outputs, and should not carry any residual influence from experts that were not chosen at all.</span>

---

## <span style="font-size: 16px;">Why the tie-breaking rule needs an explicit, guaranteed mechanism</span>

<span style="font-size: 14px;">Router logits are the output of a learned linear layer, and in principle any two experts could receive exactly equal scores, whether by coincidence, by an all-zero or otherwise degenerate router, or in constructed test cases designed to exercise this exact situation.</span>

<span style="font-size: 14px;">Without an explicit rule, "which of two equally-scored experts gets selected" is undefined, and a routing decision that varies nondeterministically between runs of the same model on the same input would make debugging, testing, and reproducing serving behavior far harder than necessary.</span>

<span style="font-size: 14px;">Defining the rule as "prefer the lower expert index" and implementing it through a genuinely guaranteed mechanism, a stable sort applied to an input already ordered by ascending expert index, rather than relying on whatever tie-breaking a particular sorting or top-k implementation happens to exhibit, is what makes the routing decision fully deterministic and reproducible for identical inputs.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">Routing runs once per token per MoE layer, and its output directly determines which experts' weights need to be loaded and which expert computations need to be dispatched, which is the subject of the very next problem in this study plan.</span>

<span style="font-size: 14px;">An incorrect or nondeterministic routing decision does not just affect representational quality; it can silently break request reproducibility, complicate debugging of production model behavior, and, in a distributed serving system where different experts may live on different devices, determine which device-to-device communication must happen for a given token.</span>

<span style="font-size: 14px;">Getting the ranking, selection, and weighting exactly right, with fully deterministic tie handling, is a prerequisite for every later problem in this section that builds on top of routing, including token dispatch, aggregation, and expert-parallel serving.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Sorting each token's row of num_experts logits costs $O(E \log E)$ using a standard comparison-based sort. Across $N$ tokens, the total routing cost is $O(N \cdot E \log E)$. The subsequent softmax over the selected top_k values costs only $O(top\_k)$ per token, negligible next to the sort whenever num_experts is reasonably large.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">The sorted intermediate tensors are the same size as the input logits, $O(N \cdot E)$. The final routing decision, expert_indices and routing_weights, is only $O(N \cdot top\_k)$, which is the size that actually needs to be communicated to whatever dispatch mechanism sends tokens to their selected experts; this is a substantial reduction whenever top_k is small relative to the total number of experts, which is the normal operating regime for sparse MoE models with many available experts, some of which are trained with dozens or hundreds of experts and route each token to only one or two of them.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Using torch.topk without accounting for its unspecified tie-breaking behavior, producing routing decisions that may not match the required lower-index-wins rule.</span>

* <span style="font-size: 14px;">Computing softmax over the full set of logits before slicing to the top-k, rather than softmax over only the selected top-k values, which produces routing weights that do not sum to 1 over the selected experts and are silently influenced by unselected experts.</span>

* <span style="font-size: 14px;">Misaligning expert_indices and routing_weights after sorting, for example sorting one tensor but not applying the identical permutation to the other, which silently pairs the wrong weight with the wrong expert.</span>

* <span style="font-size: 14px;">Assuming ties are a theoretical edge case that will not occur in practice, and therefore not testing them, when degenerate or symmetric router outputs are a realistic occurrence, particularly early in training, under quantized router weights, or in adversarially constructed inputs designed to probe routing determinism.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$Rinmathbb{R}^{T	imes E}$ is the router-logit matrix for $T$ tokens and $E$ experts, $k$ is the selected expert count, $I_{t,r}$ is the expert at route slot $r$, and $G_{t,r}$ is its normalized gate weight.</span>

<span style="font-size: 14px;">Router logits have shape $(T,E)$; selected indices and weights each have shape $(T,k)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: router_logits = [[1.0, 5.0, 2.0, 0.0]], top_k = 1</span>

<span style="font-size: 14px;">Output: expert_indices = [[1]], routing_weights = [[1.0]]</span>

<span style="font-size: 14px;">Explanation: With top_k = 1, softmax over a single value is always 1.0.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">Sparse MoE architectures such as GShard and later expert models use routing to increase parameter capacity without evaluating every expert for every token. The router output is therefore both a model decision and a systems dispatch plan.</span>

<span style="font-size: 14px;">Top-k selection and gate normalization are separate contracts. Ranking decides which experts run, while the selected softmax decides how their outputs are combined after execution.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">Stable descending ranking produces the required expert order and deterministic index tie-break.</span>

<span style="font-size: 14px;">Normalizing only selected logits gives a sparse gate whose active weights sum to one.</span>

<span style="font-size: 14px;">Returning indices and weights in matching slots preserves the association needed by dispatch and aggregation.</span>

<span style="font-size: 14px;">For $k=1$, normalization produces exactly one; when several experts tie, expert index determines the stable order.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Tests should include a unique maximum, multiple selected experts, ties at and around the selection boundary, and batches where different tokens prefer different experts.</span>

<span style="font-size: 14px;">A routing result can have correct weights but incorrect indices, so validation must compare paired index-weight slots and verify every row sum independently.</span>

---

## <span style="font-size: 16px;">Behavioral contract</span>

<span style="font-size: 14px;">Accept a two-dimensional router-logit tensor and a valid positive top-k value.</span>

<span style="font-size: 14px;">Return selected expert indices and routing weights with shape $(T,k)$.</span>

<span style="font-size: 14px;">For each token, selected indices must identify the $k$ largest router logits in descending order.</span>

<span style="font-size: 14px;">Equal router logits must be ordered by the lower expert index first.</span>

<span style="font-size: 14px;">Routing weights must be nonnegative and sum to one across the selected experts for every token.</span>

<span style="font-size: 14px;">For top-k equal to one, the selected expert must receive weight one.</span>

<span style="font-size: 14px;">The result must be deterministic and finite for every valid input.</span>

---

## <span style="font-size: 16px;">Operational cost</span>

<span style="font-size: 14px;">Full sorting costs $O(TElog E)$, while specialized top-k selection can reduce the ranking work.</span>

<span style="font-size: 14px;">Output storage is $O(Tk)$, with temporary ranking storage determined by the selection method.</span>

---

## <span style="font-size: 16px;">Inference review checklist</span>

<span style="font-size: 14px;">Begin with dimensional consistency. Router logits have shape $(T,E)$; selected indices and weights each have shape $(T,k)$. Every transformation must preserve the axes that identify independent requests, tokens, heads, channels, blocks, or routes. A result can have the expected element count and still be wrong when two semantic axes are exchanged, so named dimensions are part of the correctness proof rather than presentation detail.</span>

<span style="font-size: 14px;">Next, test the defining invariant rather than only a typical output. For $k=1$, normalization produces exactly one; when several experts tie, expert index determines the stable order. Boundary cases expose whether the implementation follows the mathematical contract or merely happens to agree on one convenient input.</span>

<span style="font-size: 14px;">Finally, separate numerical correctness from serving value. Full sorting costs $O(TElog E)$, while specialized top-k selection can reduce the ranking work. Output storage is $O(Tk)$, with temporary ranking storage determined by the selection method. The operation is useful only when its accuracy, state, and resource behavior all match the assumptions made by the surrounding inference system.</span>

---