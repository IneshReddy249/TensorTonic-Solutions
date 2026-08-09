LLM SERVING METRICS — CHEAT SHEET

=================================

TTFT — Time To First Token

  What: blank-screen wait before anything appears

  Formula: first_token_time - arrival_time

  Scope: one request

  Measures: queueing + prefill

  Ex: first 0.1, arrival 0 -&gt; TTFT = 0.1

TPOT — Time Per Output Token

  What: streaming speed AFTER the first token

  Formula: (last_token_time - first_token_time) / (N - 1)

  Scope: one request

  Measures: decode speed

  Ex: [0.1, 0.2, 0.3, 0.4], N=4 -&gt; (0.4-0.1)/3 = 0.1 s/token = 10 tok/s per user

  Notes:

    - divide by N-1: 4 tokens = 3 gaps (fence posts vs panels)

    - N = 1 -&gt; return exactly 0 (no gap exists)

    - lower = faster; 1/TPOT = tok/s per user

ITL — Inter-Token Latency

  What: each individual gap between consecutive tokens

  Formula: ts[j] - ts[j-1]   (one value per gap)

  Key fact: mean ITL == TPOT, always (middle terms telescope/cancel)

  Why it exists: real systems report p99 ITL, not the mean.

    A: 0.05, 0.05, 0.05, 0.05  -&gt; mean 0.05, p99 0.05

    B: 0.02, 0.02, 0.02, 0.14  -&gt; mean 0.05, p99 0.14  (user saw a freeze)

  Mean hides stalls. Tail catches them.

THROUGHPUT — tokens per second

  What: whole server's output rate, all requests pooled

  Formula: sum(N) / (max(last_real_token) - min(arrival))

  Scope: entire batch — NOT one user

  Ex: 6 tokens total, span 0.5 - 0 = 0.5 s -&gt; 6/0.5 = 12 tok/s

TWO THINGS THAT BREAK CODE

--------------------------

1. Padding lies. Never ts.max(dim=1). Gather at index N-1.

     last = ts.gather(1, (n-1).unsqueeze(1)).squeeze(1)

2. N=1 divides by zero. Clamp AND torch.where.

     denom = (n-1).float().clamp(min=1.0)

     tpot  = torch.where(n &gt; 1, (last-first)/denom, torch.zeros_like(first))

SCOPE TABLE

-----------

  metric      | scope         | who cares

  ------------|---------------|---------------------------

  TTFT        | one request   | user — "is it broken?"

  TPOT        | one request   | user — "why so slow?"

  p99 ITL     | one request   | you — debugging stalls

  throughput  | whole server  | whoever pays the GPU bill

THE TRADEOFF

------------

Raise batch size -&gt; throughput UP, TPOT UP (worse per user).

Same GPU, opposite directions.

Every serving config = a chosen point on that curve:

  pick a TPOT SLO -&gt; push batch size until you hit it

  -&gt; resulting throughput sets your cost per million tokens.

INTERVIEW FOLLOW-UPS

--------------------

Q: You already have TPOT, why track ITL?

A: Tail latency. Mean hides stalls; p99 exposes them.

Q: What causes ITL spikes?

A: KV cache preemption + recompute

   chunked prefill cutting into the decode batch

   rejected speculative draft tokens

Q: You say 3000 tok/s — output-only or input+output, at what concurrency?

A: Always state both. A throughput number without them is meaningless.