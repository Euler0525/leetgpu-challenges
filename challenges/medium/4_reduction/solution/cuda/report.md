# Local practice report: challenges/medium/4_reduction

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-11T08:58:51+00:00

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
| functional_7 | passed | 0.00390625 | 7.54411e-08 |
| functional_8 | passed | 5632 | 7.5105e-07 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.024576 ms |
| GPU latency p90 | 0.0267104 ms |
| GPU latency p99 | 0.0363725 ms |
| End-to-end latency p50 | 0.03435 ms |
| Host/sync overhead p50 | 0.009774 ms |
| Host/sync overhead ratio p50 | 28.4541 % |
| Call throughput | 40690.1 calls/s |
| Output element throughput | 40690.1 elements/s |
| Reference speedup | 12.4167 x |
| Reference GPU latency p50 | 0.305152 ms |
| Reference effective bandwidth p50 | 54.9799 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 16,777,216 bytes |
| Output writes | 4 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 16,777,220 bytes |
| Effective bandwidth, best | 712.348 GB/s |
| Effective bandwidth, p50 | 682.667 GB/s |
| Effective bandwidth, mean | 663.796 GB/s |
| Effective bandwidth, p90 latency | 628.116 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 896 |
| Performance max relative error | 4.27158e-07 |
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

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu124
- Triton: 3.1.0
- CUDA runtime: 12.4
- Python: 3.12.3
- Platform: Windows-11-10.0.26200-SP0
