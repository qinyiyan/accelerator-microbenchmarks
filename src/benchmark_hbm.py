"""Benchmarks HBM(High Bandwidth Memory) bandwidth.
"""

from typing import Any, Dict, Tuple

from benchmark_utils import simple_timeit
import jax
import jax.numpy as jnp


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Helper function to build the metrics and metadata for the benchmark."""
  metrics_keys = {"average_time_ms"}
  metadata = {
      key: value
      for key, value in params
      if value is not None and key not in metrics_keys
  }
  metrics = {key: value for key, value in params if key in metrics_keys}
  return metadata, metrics

def single_chip_hbm_copy(num_elements: int, dtype: jnp.dtype) -> Dict[str, Any]:
  """Benchmarks HBM with copy(read and write) on a single device."""

  def copy(a):
    return a.copy()

  a = jax.random.normal(jax.random.key(0), (num_elements,)).astype(dtype)

  jitted_f = jax.jit(copy)
  # Run once
  output = jitted_f(a)
  jax.block_until_ready(output)
  # Run the benchmark
  average_time_ms = simple_timeit(
      jitted_f,
      a,
      task="single_chip_hbm_copy",
  )
  return {"average_time_ms": average_time_ms}


def single_chip_hbm_copy_calculate_metrics(
    num_elements: int, dtype: jnp.dtype, average_time_ms: float
) -> Dict[str, Any]:
  """Calculates the metrics for the single chip hbm copy benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metadata, metrics = get_metrics_helper(params)

  # Calculate FLOPs
  tensor_size_bytes = num_elements * dtype.dtype.itemsize

  tensor_size_gbytes = (tensor_size_bytes * 2) / 10**9
  average_time_s = average_time_ms / 10**3
  bw_gbyte_sec = tensor_size_gbytes / average_time_s
  print(
      f"Tensor size: {tensor_size_bytes / 1024**2} MB, time taken (average):"
      f" {average_time_ms:.4f} ms, bandwidth: {bw_gbyte_sec:.3f} GB/s"
  )
  print()
  # Gather the metrics to report.
  metrics.update({
      "bandwidth_gbyte_sec": bw_gbyte_sec,
  })
  metrics = {key: value for key, value in metrics.items() if value is not None}
  return metadata, metrics
