# <span style="font-size: 20px;">Implement Temperature, Top-k, and Top-p Sampling</span>

<span style="font-size: 14px;">Token sampling converts model logits into deterministic testable next-token choices under temperature, top-k, and nucleus filtering controls.</span>

---

## <span style="font-size: 16px;">Why decoding needs a sampler at all</span>

<span style="font-size: 14px;">Every attention and feed-forward computation in this study plan produces logits: one unnormalized score per vocabulary token, for the next token to generate. Turning logits into an actual token choice is a separate step, and there is more than one reasonable way to do it. Always picking the highest-scoring token, greedy decoding, is simple and deterministic but tends to produce repetitive, low-diversity text, since it never considers any alternative once the top choice is fixed.</span>

<span style="font-size: 14px;">Sampling from the full probability distribution introduces diversity but risks occasionally selecting a very low-probability, incoherent token purely by chance. Temperature, top-k, and top-p are three different ways of controlling where a decoding strategy sits between these extremes, and production serving systems typically expose all three as configurable request-level parameters.</span>

$$
p = \text{softmax}(z / T)
$$

$$
token = \min\{ t : F(t) > u \}
$$

$$
F(t) = \sum_{i \leq t} p_i
$$

---

## <span style="font-size: 16px;">Why temperature is a division, not a separate formula</span>

<span style="font-size: 14px;">Dividing logits by $T$ before softmax rescales the gaps between logits: for $T < 1$, differences are amplified, and softmax produces a sharper, more peaked distribution concentrated on the highest-scoring tokens; for $T > 1$, differences are compressed, and softmax produces a flatter distribution closer to uniform.</span>

<span style="font-size: 14px;">This is a single continuous knob, with $T = 1$ recovering the model's own unmodified distribution, and it needs no special-casing except at the boundary $T = 0$, where division is undefined and the distribution's limit is a point mass on the argmax, which is why greedy decoding is implemented as a distinct, explicit branch rather than as an extreme value passed through the same division.</span>

---

## <span style="font-size: 16px;">Why top-k and top-p solve different problems</span>

<span style="font-size: 14px;">Top-k truncates the candidate set to a fixed count, regardless of how the probability mass is actually distributed among those tokens: it removes the long tail of implausible tokens, but if only two tokens are genuinely plausible, top-k with $k = 50$ still allows 48 essentially-noise tokens to remain candidates.</span>

<span style="font-size: 14px;">Top-p, nucleus sampling, instead truncates based on how much cumulative probability mass the surviving tokens actually represent, so it adapts to the shape of the distribution: a highly confident distribution keeps very few tokens, since a small number already reach the target cumulative mass, while a genuinely ambiguous distribution keeps more.</span>

<span style="font-size: 14px;">Using both together, first top-k then top-p, applies a coarse fixed-size cutoff before a mass-based refinement, which is the standard practical combination, since running top-p directly on an unfiltered distribution with a very long tail can still be computationally wasteful even though it is not incorrect.</span>

---

## <span style="font-size: 16px;">Why nucleus filtering requires sorting, but sampling does not</span>

<span style="font-size: 14px;">Determining which tokens belong in the top-p nucleus is inherently a question about probability rank: "the smallest set of tokens, taken in order of decreasing probability, whose cumulative sum first reaches top_p" cannot be answered without first ordering the tokens by probability.</span>

<span style="font-size: 14px;">Sampling from the final, already-filtered and renormalized distribution, however, does not require the tokens to be considered in probability order; the inverse-CDF method walks the cumulative sum in whatever fixed order is convenient, and natural (original vocabulary index) order is simplest, since it requires no additional bookkeeping to map back to real token IDs after the sampling decision is made.</span>

<span style="font-size: 14px;">This is why the algorithm sorts once, for the purpose of determining the nucleus boundary, and then discards the sorted order entirely once the keep-mask has been scattered back to original positions.</span>

---

## <span style="font-size: 16px;">Why the boundary condition on nucleus inclusion always keeps at least one token</span>

<span style="font-size: 14px;">The correct nucleus size can genuinely be a single token, when that token's own probability already meets or exceeds top_p. If the inclusion rule were instead "keep position $j$ if the cumulative sum up to and including $j$ is still below top_p," a single dominant token whose own probability exceeds top_p would be excluded by that rule, leaving an empty distribution.</span>

<span style="font-size: 14px;">Checking the cumulative sum *before* the current position, and unconditionally keeping the first (most probable) position regardless of that check, is what guarantees the nucleus is never empty while still producing the intuitively correct minimal set in every other case.</span>

---

## <span style="font-size: 16px;">Why the sampling rule needs a precise inequality, not just "a random number"</span>

<span style="font-size: 14px;">Inverse-CDF sampling works by treating the cumulative distribution function $F$ as a mapping from the unit interval to token indices, and drawing a uniform value $u \in [0, 1)$ then finding where it falls. What happens when $u$ lands exactly on a cumulative boundary, meaning exactly at the point where one token's probability mass ends and the next begins, is not something probability theory alone answers; it requires a concrete implementation convention.</span>

<span style="font-size: 14px;">Defining the sampled token as the smallest $t$ with $F(t) > u$, a strict inequality, means a draw exactly equal to a boundary value deterministically resolves to the token immediately after that boundary, not the one whose mass ends there.</span>

<span style="font-size: 14px;">This is precisely why supplying uniform_draws explicitly, rather than calling an uncontrolled random number generator, matters for this problem: it makes an otherwise probabilistic algorithm fully deterministic and exactly reproducible given the same inputs, which is essential both for testing and for any serving system that needs deterministic replay of a specific generation.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">Sampling is executed once per generated token, for every request being served, making it one of the most frequently invoked operations in the entire serving stack, even though each individual call is cheap relative to the transformer forward pass that produces the logits.</span>

<span style="font-size: 14px;">Supporting temperature, top-k, and top-p as independent, composable controls, rather than a single fixed decoding strategy, is what lets a serving system expose per-request sampling behavior to callers without needing separate code paths for each configuration: disabling top-k and top-p reduces to plain temperature sampling, and setting temperature to 0 reduces to greedy decoding, all through the same function.</span>

<span style="font-size: 14px;">Because the entire computation depends only on the current step's logits and an externally supplied random draw, it composes cleanly with speculative decoding, covered later in this plan, where a draft model's proposed tokens must be verified against exactly this kind of probability computation.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Top-k selection costs $O(V \log V)$ per row using a comparison-based topk, or $O(V)$ with a selection algorithm; top-p sorting costs $O(V \log V)$ per row. Softmax, cumulative sum, and the final search are each $O(V)$ per row. Across a batch of size $B$, the total cost is $O(B \cdot V \log V)$, dominated by whichever sorting step is used, and independent of how many tokens have already been generated.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">Every tensor involved, logits, scaled, probs, the sorted intermediates, and the cumulative sums, is $O(B \cdot V)$, proportional only to the vocabulary size and batch size, with no dependence on sequence length or how many tokens have already been generated. This is why sampling cost stays constant across a long decoding run, in contrast to attention itself, whose cost and memory grow with the number of previously generated tokens.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Applying top-p filtering to raw logits instead of softmax probabilities, which makes the "cumulative probability" threshold meaningless, since logits are not a probability distribution.</span>

* <span style="font-size: 14px;">Using a non-strict cumulative sum comparison for nucleus inclusion, which can either exclude a single dominant token that alone exceeds top_p, or include one token too many depending on which boundary convention is chosen inconsistently.</span>

* <span style="font-size: 14px;">Forgetting to renormalize after masking out filtered tokens, leaving the cumulative distribution short of 1 and causing high draws to fail to select any token.</span>

* <span style="font-size: 14px;">Sampling in sorted-by-probability order and forgetting to map the result back to the original token index, silently returning the wrong token ID.</span>

* <span style="font-size: 14px;">Treating temperature = 0 as an ordinary small number to divide by, rather than a distinct greedy branch, risking overflow or NaN instead of a clean argmax.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$z_j$ is the logit for vocabulary item $j$, $T$ is temperature, $k$ is the top-k limit, $p$ is the nucleus mass threshold, and $uin[0,1)$ is the supplied uniform draw.</span>

<span style="font-size: 14px;">Logits have shape $(B,V)$, uniform draws have shape $(B)$, and returned token IDs have shape $(B)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: logits = [[1.0, 3.0, 2.0, 0.5]], temperature = 0, top_k = 0, top_p = 1.0, uniform_draws = [0.5]</span>

<span style="font-size: 14px;">Output: tensor([1])</span>

<span style="font-size: 14px;">Explanation: Temperature 0 always picks the argmax, index 1, regardless of the draw.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">Sampling is a decoding policy rather than a change to the transformer itself. Temperature controls concentration, top-k bounds candidate count, and top-p adapts candidate count to the probability distribution at each step.</span>

<span style="font-size: 14px;">Serving systems need deterministic replay for tests and incident analysis. Supplying the uniform draws separates distribution construction from randomness, making every boundary and filtering rule directly verifiable.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">Temperature changes the relative logit scale before probabilities are formed, while the zero-temperature branch implements the greedy limit explicitly.</span>

<span style="font-size: 14px;">Top-k and top-p remove disallowed tokens before the remaining mass is renormalized.</span>

<span style="font-size: 14px;">Inverse-CDF selection maps each supplied uniform draw to exactly one token without an uncontrolled random-number generator.</span>

<span style="font-size: 14px;">At least one token always survives filtering, and a draw at a cumulative boundary follows the contract consistently.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Validation should cover the greedy limit, both disabled-filter conventions, equal logits, a draw just below and exactly on a cumulative boundary, and cases where top-k and top-p remove different tokens.</span>

<span style="font-size: 14px;">The final probabilities should sum to one over surviving tokens, and every filtered token should have exactly zero probability before inverse-CDF selection.</span>

---