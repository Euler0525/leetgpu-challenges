# Local practice report: challenges/easy/3_matrix_transpose

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-08T10:43:36+00:00

## Correctness

Passed 13/13 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 0 | 0 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 0 | 0 |
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 0 | 0 |
| functional_9 | passed | 0 | 0 |
| functional_10 | passed | 0 | 0 |
| functional_11 | passed | 0 | 0 |
| functional_12 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 1.7801 ms |
| GPU latency p90 | 1.83637 ms |
| GPU latency p99 | 1.84696 ms |
| End-to-end latency p50 | 1.7947 ms |
| Host/sync overhead p50 | 0.0146039 ms |
| Host/sync overhead ratio p50 | 0.813726 % |
| Call throughput | 561.767 calls/s |
| Output element throughput | 2.35942e+10 elements/s |
| Reference speedup | 1.66592 x |
| Reference GPU latency p50 | 2.9655 ms |
| Reference effective bandwidth p50 | 113.303 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 168,000,000 bytes |
| Output writes | 168,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 336,000,000 bytes |
| Effective bandwidth, best | 189.887 GB/s |
| Effective bandwidth, p50 | 188.754 GB/s |
| Effective bandwidth, mean | 187.468 GB/s |
| Effective bandwidth, p90 latency | 182.97 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 336,000,000 bytes |
| Resident PyTorch reservation | 2,906,652,672 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 336,000,000 bytes |
| Peak PyTorch reservation | 2,906,652,672 bytes |

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
