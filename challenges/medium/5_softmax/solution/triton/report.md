# Local practice report: challenges/medium/5_softmax

- Backend: `triton`
- Status: **passed**
- Updated: 2026-09-09T01:08:55+00:00

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
| functional_9 | passed | 4.65661e-10 | 5.11173e-07 |
| functional_10 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.500656 ms |
| GPU latency p90 | 0.503808 ms |
| GPU latency p99 | 0.504832 ms |
| End-to-end latency p50 | 0.5306 ms |
| Host/sync overhead p50 | 0.029944 ms |
| Host/sync overhead ratio p50 | 5.64342 % |
| Call throughput | 1997.38 calls/s |
| Output element throughput | 9.9869e+08 elements/s |
| Reference speedup | 0.199482 x |
| Reference GPU latency p50 | 0.099872 ms |
| Reference effective bandwidth p50 | 40.0513 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 2,000,000 bytes |
| Output writes | 2,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 4,000,000 bytes |
| Effective bandwidth, best | 8.088 GB/s |
| Effective bandwidth, p50 | 7.98952 GB/s |
| Effective bandwidth, mean | 7.9991 GB/s |
| Effective bandwidth, p90 latency | 7.93953 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 1.09139e-11 |
| Performance max relative error | 9.29619e-07 |
| Resident PyTorch allocation | 4,000,768 bytes |
| Resident PyTorch reservation | 44,040,192 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 4,000,768 bytes |
| Peak PyTorch reservation | 44,040,192 bytes |

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
