# Local practice report: challenges/easy/66_rgb_to_grayscale

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T10:50:02+00:00

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
| functional_7 | passed | 1.52588e-05 | 9.28274e-08 |
| functional_8 | passed | 1.52588e-05 | 1.08477e-07 |
| functional_9 | passed | 1.52588e-05 | 1.15128e-07 |
| functional_10 | passed | 3.05176e-05 | 1.85509e-07 |
| functional_11 | passed | 3.05176e-05 | 2.09376e-07 |
| functional_12 | passed | 3.05176e-05 | 1.88253e-07 |
| functional_13 | passed | 1.52588e-05 | 8.68749e-08 |
| functional_14 | passed | 1.52588e-05 | 7.27205e-08 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.94616 ms |
| GPU latency p90 | 0.95617 ms |
| GPU latency p99 | 0.958579 ms |
| End-to-end latency p50 | 0.96005 ms |
| Host/sync overhead p50 | 0.01389 ms |
| Host/sync overhead ratio p50 | 1.4468 % |
| Call throughput | 1056.9 calls/s |
| Output element throughput | 4.43298e+09 elements/s |
| Reference speedup | 1.64224 x |
| Reference GPU latency p50 | 1.55382 ms |
| Reference effective bandwidth p50 | 43.1895 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 50,331,648 bytes |
| Output writes | 16,777,216 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 67,108,864 bytes |
| Effective bandwidth, best | 71.2348 GB/s |
| Effective bandwidth, p50 | 70.9276 GB/s |
| Effective bandwidth, mean | 70.8492 GB/s |
| Effective bandwidth, p90 latency | 70.1851 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 3.05176e-05 |
| Performance max relative error | 2.33415e-07 |
| Resident PyTorch allocation | 75,628,544 bytes |
| Resident PyTorch reservation | 312,475,648 bytes |
| Incremental peak allocation | 16,777,728 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 92,406,272 bytes |
| Peak PyTorch reservation | 312,475,648 bytes |

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
