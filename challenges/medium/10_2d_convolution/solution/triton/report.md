# Local practice report: challenges/medium/10_2d_convolution

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-31T03:29:13+00:00

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
| GPU latency p50 | 3.29523 ms |
| GPU latency p90 | 3.57714 ms |
| GPU latency p99 | 3.59752 ms |
| End-to-end latency p50 | 3.31785 ms |
| Host/sync overhead p50 | 0.0226179 ms |
| Host/sync overhead ratio p50 | 0.681705 % |
| Call throughput | 303.469 calls/s |
| Output element throughput | 2.83785e+09 elements/s |
| Reference speedup | 8.80622 x |
| Reference GPU latency p50 | 29.0185 ms |
| Reference effective bandwidth p50 | 2.5899 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 37,749,636 bytes |
| Output writes | 37,405,456 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 75,155,092 bytes |
| Effective bandwidth, best | 22.9451 GB/s |
| Effective bandwidth, p50 | 22.8072 GB/s |
| Effective bandwidth, mean | 22.1416 GB/s |
| Effective bandwidth, p90 latency | 21.0098 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 75,498,496 bytes |
| Resident PyTorch reservation | 643,825,664 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 75,498,496 bytes |
| Peak PyTorch reservation | 643,825,664 bytes |

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
