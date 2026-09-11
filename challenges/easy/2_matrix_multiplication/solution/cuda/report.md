# Local practice report: challenges/easy/2_matrix_multiplication

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-09-11T09:36:26+00:00

## Correctness

Passed 15/15 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 0 | 0 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 3.8147e-06 | 1.36256e-07 |
| functional_7 | passed | 3.05176e-05 | 5.88783e-07 |
| functional_8 | passed | 6.10352e-05 | 1.90059e-05 |
| functional_9 | passed | 3.05176e-05 | 8.95202e-06 |
| functional_10 | passed | 4.57764e-05 | 7.01724e-06 |
| functional_11 | passed | 0 | 0 |
| functional_12 | passed | 2.98023e-08 | 1.19492e-07 |
| functional_13 | passed | 1.49012e-08 | 1.81957e-07 |
| functional_14 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 391.042 ms |
| GPU latency p90 | 410.314 ms |
| GPU latency p99 | 439.86 ms |
| End-to-end latency p50 | 391.081 ms |
| Host/sync overhead p50 | 0.038551 ms |
| Host/sync overhead ratio p50 | 0.00985756 % |
| Call throughput | 2.55727 calls/s |
| Output element throughput | 8.58077e+07 elements/s |
| Reference speedup | 0.156548 x |
| Reference GPU latency p50 | 61.2167 ms |
| Reference effective bandwidth p50 | 7.12563 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 301,989,888 bytes |
| Output writes | 134,217,728 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 436,207,616 bytes |
| Effective bandwidth, best | 1.12468 GB/s |
| Effective bandwidth, p50 | 1.1155 GB/s |
| Effective bandwidth, mean | 1.10068 GB/s |
| Effective bandwidth, p90 latency | 1.06311 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 444,727,296 bytes |
| Resident PyTorch reservation | 2,506,096,640 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 444,727,296 bytes |
| Peak PyTorch reservation | 2,506,096,640 bytes |

Warmup iterations: 5; measured iterations: 30.

Effective memory bandwidth uses the minimum bytes implied by input/output arguments, not hardware-counter DRAM traffic. PyTorch allocator peaks do not include allocations made directly inside native CUDA code.

## Profiler

Chrome trace: `profile_trace.json`

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.13.0+cu130
- Triton: 3.7.1
- CUDA runtime: 13.0
- Python: 3.14.6
- Platform: Windows-11-10.0.26200-SP0
