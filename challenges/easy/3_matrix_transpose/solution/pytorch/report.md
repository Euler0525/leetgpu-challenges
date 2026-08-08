# Local practice report: challenges/easy/3_matrix_transpose

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-08T07:41:15+00:00

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
| GPU latency p50 | 2.9527 ms |
| GPU latency p90 | 2.99412 ms |
| GPU latency p99 | 3.04905 ms |
| End-to-end latency p50 | 2.9886 ms |
| Host/sync overhead p50 | 0.0358959 ms |
| Host/sync overhead ratio p50 | 1.2011 % |
| Call throughput | 338.673 calls/s |
| Output element throughput | 1.42242e+10 elements/s |
| Reference speedup | 0.998943 x |
| Reference GPU latency p50 | 2.94958 ms |
| Reference effective bandwidth p50 | 113.914 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 168,000,000 bytes |
| Output writes | 168,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 336,000,000 bytes |
| Effective bandwidth, best | 114.296 GB/s |
| Effective bandwidth, p50 | 113.794 GB/s |
| Effective bandwidth, mean | 113.318 GB/s |
| Effective bandwidth, p90 latency | 112.22 GB/s |
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
