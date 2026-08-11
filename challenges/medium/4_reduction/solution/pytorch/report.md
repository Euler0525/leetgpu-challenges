# Local practice report: challenges/medium/4_reduction

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-11T08:59:24+00:00

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
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.03584 ms |
| GPU latency p90 | 0.0379904 ms |
| GPU latency p99 | 0.0539491 ms |
| End-to-end latency p50 | 0.04795 ms |
| Host/sync overhead p50 | 0.01211 ms |
| Host/sync overhead ratio p50 | 25.2555 % |
| Call throughput | 27901.8 calls/s |
| Output element throughput | 27901.8 elements/s |
| Reference speedup | 8.45714 x |
| Reference GPU latency p50 | 0.303104 ms |
| Reference effective bandwidth p50 | 55.3514 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 16,777,216 bytes |
| Output writes | 4 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 16,777,220 bytes |
| Effective bandwidth, best | 512 GB/s |
| Effective bandwidth, p50 | 468.114 GB/s |
| Effective bandwidth, mean | 454.638 GB/s |
| Effective bandwidth, p90 latency | 441.617 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 16,777,728 bytes |
| Resident PyTorch reservation | 62,914,560 bytes |
| Incremental peak allocation | 1,536 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 16,779,264 bytes |
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
