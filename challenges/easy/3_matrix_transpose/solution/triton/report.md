# Local practice report: challenges/easy/3_matrix_transpose

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-08T10:27:56+00:00

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
| GPU latency p50 | 42.7305 ms |
| GPU latency p90 | 42.8014 ms |
| GPU latency p99 | 42.8418 ms |
| End-to-end latency p50 | 42.783 ms |
| Host/sync overhead p50 | 0.0524526 ms |
| Host/sync overhead ratio p50 | 0.122602 % |
| Call throughput | 23.4025 calls/s |
| Output element throughput | 9.82905e+08 elements/s |
| Reference speedup | 0.0688962 x |
| Reference GPU latency p50 | 2.94397 ms |
| Reference effective bandwidth p50 | 114.132 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 168,000,000 bytes |
| Output writes | 168,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 336,000,000 bytes |
| Effective bandwidth, best | 7.88213 GB/s |
| Effective bandwidth, p50 | 7.86324 GB/s |
| Effective bandwidth, mean | 7.86335 GB/s |
| Effective bandwidth, p90 latency | 7.8502 GB/s |
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
