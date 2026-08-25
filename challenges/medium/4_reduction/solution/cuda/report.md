# Local practice report: challenges/medium/4_reduction

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-25T11:42:39+00:00

## Correctness

Passed 9/9 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 0 | 0 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 0 | 0 |
| functional_7 | passed | 0.0078125 | 1.50882e-07 |
| functional_8 | passed | 1536 | 2.04832e-07 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.044032 ms |
| GPU latency p90 | 0.048128 ms |
| GPU latency p99 | 0.0696013 ms |
| End-to-end latency p50 | 0.05675 ms |
| Host/sync overhead p50 | 0.012718 ms |
| Host/sync overhead ratio p50 | 22.4106 % |
| Call throughput | 22710.8 calls/s |
| Output element throughput | 22710.8 elements/s |
| Reference speedup | 7.01163 x |
| Reference GPU latency p50 | 0.308736 ms |
| Reference effective bandwidth p50 | 54.3416 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 16,777,216 bytes |
| Output writes | 4 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 16,777,220 bytes |
| Effective bandwidth, best | 409.6 GB/s |
| Effective bandwidth, p50 | 381.023 GB/s |
| Effective bandwidth, mean | 365.816 GB/s |
| Effective bandwidth, p90 latency | 348.596 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 768 |
| Performance max relative error | 3.66135e-07 |
| Resident PyTorch allocation | 16,777,728 bytes |
| Resident PyTorch reservation | 62,914,560 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 16,777,728 bytes |
| Peak PyTorch reservation | 62,914,560 bytes |

Warmup iterations: 5; measured iterations: 30.

Effective memory bandwidth uses the minimum bytes implied by input/output arguments, not hardware-counter DRAM traffic. PyTorch allocator peaks do not include allocations made directly inside native CUDA code.

## Profiler

Chrome trace: `profile_trace.json`

## Native Nsight profiles

| Tool | Status | File | Size | Version | Metric set | Source embedded |
|---|---:|---|---:|---|---|---:|
| nsys | stale | `solution.nsys-rep` | 56,723 bytes | NVIDIA Nsight Systems version 2026.1.3.243-261337792075v0 | n/a | no |

Open nsys: `nsys-ui solution.nsys-rep`
| ncu | stale | `solution.ncu-rep` | 224,502 bytes | NVIDIA (R) Nsight Compute Command Line Profiler Copyright (c) 2018-2026 NVIDIA Corporation Version 2026.2.0.0 (build 37790515) (public-release) | detailed | yes |

Open ncu: `ncu-ui solution.ncu-rep`

A stale report was generated from a different solution SHA-256 and was left on disk intentionally.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.13.0+cu130
- Triton: 3.7.1
- CUDA runtime: 13.0
- Python: 3.14.6
- Platform: Windows-11-10.0.26200-SP0
