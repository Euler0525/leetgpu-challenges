# Local practice report: challenges/medium/6_softmax_attention

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-13T01:48:32+00:00

## Correctness

Passed 5/5 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 4.76837e-07 | 1.11181e-07 |
| functional_1 | passed | 4.76837e-07 | 1.11181e-07 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 5.91908e-42 | 7.56516e-06 |
| functional_4 | passed | 7.91624e-09 | 1.36854e-05 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.354912 ms |
| GPU latency p90 | 0.445024 ms |
| GPU latency p99 | 2.05135 ms |
| End-to-end latency p50 | 0.38615 ms |
| Host/sync overhead p50 | 0.031238 ms |
| Host/sync overhead ratio p50 | 8.0896 % |
| Call throughput | 2817.6 calls/s |
| Output element throughput | 1.84654e+08 elements/s |
| Reference speedup | 0.46141 x |
| Reference GPU latency p50 | 0.16376 ms |
| Reference effective bandwidth p50 | 4.80234 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 524,288 bytes |
| Output writes | 262,144 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 786,432 bytes |
| Effective bandwidth, best | 2.37771 GB/s |
| Effective bandwidth, p50 | 2.21585 GB/s |
| Effective bandwidth, mean | 1.72097 GB/s |
| Effective bandwidth, p90 latency | 1.76717 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 8.3819e-09 |
| Performance max relative error | 0.0802485 |
| Resident PyTorch allocation | 9,306,112 bytes |
| Resident PyTorch reservation | 25,165,824 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 9,306,112 bytes |
| Peak PyTorch reservation | 25,165,824 bytes |

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
