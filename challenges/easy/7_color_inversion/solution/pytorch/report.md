# Local practice report: challenges/easy/7_color_inversion

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-09T04:22:59+00:00

## Correctness

Passed 8/8 cases.

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

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 1.80509 ms |
| GPU latency p90 | 1.86487 ms |
| GPU latency p99 | 1.90468 ms |
| End-to-end latency p50 | 1.81685 ms |
| Host/sync overhead p50 | 0.011762 ms |
| Host/sync overhead ratio p50 | 0.647382 % |
| Call throughput | 553.99 calls/s |
| Output element throughput | 4.6472e+10 elements/s |
| Reference speedup | 1.07301 x |
| Reference GPU latency p50 | 1.93688 ms |
| Reference effective bandwidth p50 | 86.6198 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 0 bytes |
| Output writes | 0 bytes |
| In-place reads | 83,886,080 bytes |
| In-place writes | 83,886,080 bytes |
| Minimum total traffic | 167,772,160 bytes |
| Effective bandwidth, best | 93.3196 GB/s |
| Effective bandwidth, p50 | 92.944 GB/s |
| Effective bandwidth, mean | 92.1632 GB/s |
| Effective bandwidth, p90 latency | 89.9644 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 167,772,160 bytes |
| Resident PyTorch reservation | 2,665,480,192 bytes |
| Incremental peak allocation | 62,914,560 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 230,686,720 bytes |
| Peak PyTorch reservation | 2,665,480,192 bytes |

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
