# Local practice report: challenges/medium/5_softmax

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-11T13:00:58+00:00

## Correctness

Passed 11/11 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 5.96046e-08 | 8.95986e-08 |
| functional_1 | passed | 5.96046e-08 | 8.95986e-08 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 5.96046e-08 | 8.95986e-08 |
| functional_4 | passed | 5.96046e-08 | 2.18879e-07 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 4.65661e-10 | 8.06754e-08 |
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 0 | 0 |
| functional_9 | passed | 4.65661e-10 | 5.84198e-07 |
| functional_10 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.090912 ms |
| GPU latency p90 | 0.101475 ms |
| GPU latency p99 | 0.103286 ms |
| End-to-end latency p50 | 0.1034 ms |
| Host/sync overhead p50 | 0.012488 ms |
| Host/sync overhead ratio p50 | 12.0774 % |
| Call throughput | 10999.6 calls/s |
| Output element throughput | 5.49982e+09 elements/s |
| Reference speedup | 1.05245 x |
| Reference GPU latency p50 | 0.09568 ms |
| Reference effective bandwidth p50 | 41.806 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 2,000,000 bytes |
| Output writes | 2,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 4,000,000 bytes |
| Effective bandwidth, best | 54.2535 GB/s |
| Effective bandwidth, p50 | 43.9986 GB/s |
| Effective bandwidth, mean | 44.475 GB/s |
| Effective bandwidth, p90 latency | 39.4185 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 7.27596e-12 |
| Performance max relative error | 1.01413e-06 |
| Resident PyTorch allocation | 4,000,768 bytes |
| Resident PyTorch reservation | 44,040,192 bytes |
| Incremental peak allocation | 1,024 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 4,001,792 bytes |
| Peak PyTorch reservation | 44,040,192 bytes |

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
