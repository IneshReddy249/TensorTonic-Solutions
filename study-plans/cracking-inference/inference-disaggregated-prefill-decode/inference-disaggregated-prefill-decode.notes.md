Here is the complete timeline of how a request moves through a disaggregated serving setup (where prefill and decode happen on separate machines), formatted in a simple, copy-pasteable note:

Plaintext

```
# Disaggregated Serving Lifecycle: Prefill to Decode

--- STAGE 1: PREFILL (Reading the Prompt) ---
start_p  = max(arrival_i, avail[p])
end_i    = start_p + P_i / rate_p
avail[p] = end_i

* How it works: The prefill machine (p) starts as soon as it's free AND the user's request has arrived. It chunks through the whole prompt (P_i) at its processing speed. Finally, it updates its own schedule to say it's busy until the job finishes (end_i).

--- STAGE 2: KV TRANSFER (Moving the Data) ---
kv_ready_i = end_i + (P_i / bandwidth) + latency

* How it works: You can't generate words without context. This calculates exactly when the prefill machine finishes sending the KV cache data over the network to the decode machine. It adds the physical transmission time (data size / network speed) plus a flat network delay (latency).

--- STAGE 3: DECODE (Generating the Answer) ---
start_d  = max(kv_ready_i, avail[d])
ttft_i   = start_d + 1 / rate_d
comp_i   = start_d + D_i / rate_d
avail[d] = comp_i

* How it works: The decode machine (d) starts as soon as it's free AND all the KV data has arrived from Stage 2. 
  - Time To First Token (ttft_i) happens after it generates just 1 token. 
  - Completion (comp_i) happens after it generates all requested tokens (D_i). 
  - It then blocks off its schedule until the full generation is done.

```