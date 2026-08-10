# Local practice report: challenges/easy/52_silu

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-10T08:19:46+00:00

## Correctness

Passed 8/8 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 1.19209e-07 | 8.1532e-08 |
| functional_1 | passed | 5.96046e-08 | 8.1532e-08 |
| functional_2 | passed | 1.19209e-07 | 6.76713e-08 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 1.49012e-08 | 6.25033e-08 |
| functional_5 | passed | 5.96046e-08 | 8.1532e-08 |
| functional_6 | passed | 4.76837e-07 | 3.76683e-06 |
| functional_7 | passed | 9.53674e-07 | 3.27285e-06 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.02448 ms |
| GPU latency p90 | 0.0356128 ms |
| GPU latency p99 | 0.036864 ms |
| End-to-end latency p50 | 0.0365 ms |
| Host/sync overhead p50 | 0.01202 ms |
| Host/sync overhead ratio p50 | 32.9315 % |
| Call throughput | 40849.7 calls/s |
| Output element throughput | 2.04248e+09 elements/s |
| Reference speedup | 1.55033 x |
| Reference GPU latency p50 | 0.037952 ms |
| Reference effective bandwidth p50 | 10.5396 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000 bytes |
| Output writes | 200,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000 bytes |
| Effective bandwidth, best | 20.8333 GB/s |
| Effective bandwidth, p50 | 16.3399 GB/s |
| Effective bandwidth, mean | 14.7088 GB/s |
| Effective bandwidth, p90 latency | 11.2319 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 1.43051e-06 |
| Performance max relative error | 3.37149e-06 |
| Resident PyTorch allocation | 400,384 bytes |
| Resident PyTorch reservation | 4,194,304 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 400,384 bytes |
| Peak PyTorch reservation | 4,194,304 bytes |

Warmup iterations: 5; measured iterations: 30.

Effective memory bandwidth uses the minimum bytes implied by input/output arguments, not hardware-counter DRAM traffic. PyTorch allocator peaks do not include allocations made directly inside native CUDA code.

## Profiler

Chrome trace: `profile_trace.json`

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu124
- Triton: 3.1.0
- CUDA runtime: 12.4
- Python: 3.12.3
- Platform: Windows-11-10.0.26200-SP0
