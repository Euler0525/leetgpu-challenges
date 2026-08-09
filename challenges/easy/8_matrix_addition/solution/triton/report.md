# Local practice report: challenges/easy/8_matrix_addition

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-09T06:06:42+00:00

## Correctness

Passed 12/12 cases.

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

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.877568 ms |
| GPU latency p90 | 0.885658 ms |
| GPU latency p99 | 0.924518 ms |
| End-to-end latency p50 | 0.88925 ms |
| Host/sync overhead p50 | 0.011682 ms |
| Host/sync overhead ratio p50 | 1.31369 % |
| Call throughput | 1139.51 calls/s |
| Output element throughput | 1.91179e+10 elements/s |
| Reference speedup | 1.00875 x |
| Reference GPU latency p50 | 0.885248 ms |
| Reference effective bandwidth p50 | 227.424 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 134,217,728 bytes |
| Output writes | 67,108,864 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 201,326,592 bytes |
| Effective bandwidth, best | 231.304 GB/s |
| Effective bandwidth, p50 | 229.414 GB/s |
| Effective bandwidth, mean | 228.818 GB/s |
| Effective bandwidth, p90 latency | 227.319 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 201,326,592 bytes |
| Resident PyTorch reservation | 1,210,056,704 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 201,326,592 bytes |
| Peak PyTorch reservation | 1,210,056,704 bytes |

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
