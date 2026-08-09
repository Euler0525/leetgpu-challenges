# Local practice report: challenges/easy/2_matrix_multiplication

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-09T07:21:00+00:00

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
| functional_6 | passed | 0 | 0 |
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 0 | 0 |
| functional_9 | passed | 0 | 0 |
| functional_10 | passed | 0 | 0 |
| functional_11 | passed | 0 | 0 |
| functional_12 | passed | 0 | 0 |
| functional_13 | passed | 0 | 0 |
| functional_14 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 59.9971 ms |
| GPU latency p90 | 61.2927 ms |
| GPU latency p99 | 61.9919 ms |
| End-to-end latency p50 | 60.0649 ms |
| Host/sync overhead p50 | 0.0677606 ms |
| Host/sync overhead ratio p50 | 0.112812 % |
| Call throughput | 16.6675 calls/s |
| Output element throughput | 5.59268e+08 elements/s |
| Reference speedup | 1.04168 x |
| Reference GPU latency p50 | 62.4977 ms |
| Reference effective bandwidth p50 | 6.97958 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 301,989,888 bytes |
| Output writes | 134,217,728 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 436,207,616 bytes |
| Effective bandwidth, best | 7.4738 GB/s |
| Effective bandwidth, p50 | 7.27048 GB/s |
| Effective bandwidth, mean | 7.26478 GB/s |
| Effective bandwidth, p90 latency | 7.11679 GB/s |
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
- PyTorch: 2.5.1+cu124
- Triton: 3.1.0
- CUDA runtime: 12.4
- Python: 3.12.3
- Platform: Windows-11-10.0.26200-SP0
