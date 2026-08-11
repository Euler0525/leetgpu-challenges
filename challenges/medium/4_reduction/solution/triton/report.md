# Local practice report: challenges/medium/4_reduction

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-11T02:28:29+00:00

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
| functional_8 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.04888 ms |
| GPU latency p90 | 0.0696416 ms |
| GPU latency p99 | 0.0793462 ms |
| End-to-end latency p50 | 0.0626 ms |
| Host/sync overhead p50 | 0.01372 ms |
| Host/sync overhead ratio p50 | 21.9169 % |
| Call throughput | 20458.3 calls/s |
| Output element throughput | 20458.3 elements/s |
| Reference speedup | 6.29525 x |
| Reference GPU latency p50 | 0.307712 ms |
| Reference effective bandwidth p50 | 54.5225 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 16,777,216 bytes |
| Output writes | 4 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 16,777,220 bytes |
| Effective bandwidth, best | 409.92 GB/s |
| Effective bandwidth, p50 | 343.233 GB/s |
| Effective bandwidth, mean | 309.564 GB/s |
| Effective bandwidth, p90 latency | 240.908 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 16,777,728 bytes |
| Resident PyTorch reservation | 62,914,560 bytes |
| Incremental peak allocation | 1,024 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 16,778,752 bytes |
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
