# Local practice report: challenges/easy/21_relu

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-10T06:02:06+00:00

## Correctness

Passed 12/12 cases.

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

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.898176 ms |
| GPU latency p90 | 0.93305 ms |
| GPU latency p99 | 0.950863 ms |
| End-to-end latency p50 | 0.91005 ms |
| Host/sync overhead p50 | 0.011874 ms |
| Host/sync overhead ratio p50 | 1.30477 % |
| Call throughput | 1113.37 calls/s |
| Output element throughput | 2.78342e+10 elements/s |
| Reference speedup | 2.27448 x |
| Reference GPU latency p50 | 2.04288 ms |
| Reference effective bandwidth p50 | 97.901 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 100,000,000 bytes |
| Output writes | 100,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 200,000,000 bytes |
| Effective bandwidth, best | 225.055 GB/s |
| Effective bandwidth, p50 | 222.674 GB/s |
| Effective bandwidth, mean | 221.058 GB/s |
| Effective bandwidth, p90 latency | 214.351 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 201,326,592 bytes |
| Resident PyTorch reservation | 1,713,373,184 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 201,326,592 bytes |
| Peak PyTorch reservation | 1,713,373,184 bytes |

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
