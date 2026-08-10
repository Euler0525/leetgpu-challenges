# Local practice report: challenges/easy/23_leaky_relu

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-10T07:08:38+00:00

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
| functional_6 | passed | 3.72529e-09 | 7.45058e-08 |
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 5.96046e-08 | 1.1919e-07 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 1.68747 ms |
| GPU latency p90 | 1.7367 ms |
| GPU latency p99 | 1.78204 ms |
| End-to-end latency p50 | 1.7032 ms |
| Host/sync overhead p50 | 0.015728 ms |
| Host/sync overhead ratio p50 | 0.923439 % |
| Call throughput | 592.602 calls/s |
| Output element throughput | 2.96301e+10 elements/s |
| Reference speedup | 4.98689 x |
| Reference GPU latency p50 | 8.41523 ms |
| Reference effective bandwidth p50 | 47.5329 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 200,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000,000 bytes |
| Effective bandwidth, best | 239.207 GB/s |
| Effective bandwidth, p50 | 237.041 GB/s |
| Effective bandwidth, mean | 235.458 GB/s |
| Effective bandwidth, p90 latency | 230.321 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 9.53674e-07 |
| Performance max relative error | 1.19209e-07 |
| Resident PyTorch allocation | 400,000,000 bytes |
| Resident PyTorch reservation | 3,414,163,456 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 400,000,000 bytes |
| Peak PyTorch reservation | 3,414,163,456 bytes |

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
