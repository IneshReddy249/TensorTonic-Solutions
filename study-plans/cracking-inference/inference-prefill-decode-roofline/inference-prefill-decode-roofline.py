import torch

def analyze_prefill_decode_roofline(
    d_model: int,
    num_layers: int,
    batch_size: int,
    prompt_length: int,
    generated_length: int,
    element_size: int,
    peak_flops: float,
    memory_bandwidth: float,

) -> torch.Tensor:

  params = 12 * num_layers * d_model ** 2

  prefill_flops = 2 * params * prompt_length * batch_size
  prefill_bytes = params * element_size
  prefill_ai = prefill_flops / prefill_bytes
  prefill_attainable = min(peak_flops, memory_bandwidth * prefill_ai)
  prefill_latency = prefill_flops / prefill_attainable


  decode_flops = 2 * params * batch_size * generated_length
  decode_bytes = params * element_size * generated_length
  decode_ai = decode_flops / decode_bytes
  decode_attainable = min(peak_flops, memory_bandwidth * decode_ai)
  decode_latency = decode_flops / decode_attainable

  return torch.tensor([
      [prefill_flops, prefill_bytes, prefill_ai, prefill_attainable, prefill_latency],
      [decode_flops, decode_bytes, decode_ai, decode_attainable, decode_latency]
  ], dtype=torch.float64)