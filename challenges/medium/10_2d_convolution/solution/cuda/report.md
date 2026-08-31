# Local practice report: challenges/medium/10_2d_convolution

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-31T05:50:47+00:00

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
| GPU latency p50 | 2.99288 ms |
| GPU latency p90 | 3.40183 ms |
| GPU latency p99 | 3.41222 ms |
| End-to-end latency p50 | 3.0462 ms |
| Host/sync overhead p50 | 0.05332 ms |
| Host/sync overhead ratio p50 | 1.75038 % |
| Call throughput | 334.126 calls/s |
| Output element throughput | 3.12454e+09 elements/s |
| Reference speedup | 9.82607 x |
| Reference GPU latency p50 | 29.4083 ms |
| Reference effective bandwidth p50 | 2.55558 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 37,749,636 bytes |
| Output writes | 37,405,456 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 75,155,092 bytes |
| Effective bandwidth, best | 25.4574 GB/s |
| Effective bandwidth, p50 | 25.1113 GB/s |
| Effective bandwidth, mean | 23.8855 GB/s |
| Effective bandwidth, p90 latency | 22.0925 GB/s |
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
