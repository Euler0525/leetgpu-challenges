# Local practice report: challenges/medium/10_2d_convolution

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-31T02:57:59+00:00

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
| GPU latency p50 | 28.9157 ms |
| GPU latency p90 | 30.2199 ms |
| GPU latency p99 | 30.5305 ms |
| End-to-end latency p50 | 28.9609 ms |
| Host/sync overhead p50 | 0.0452376 ms |
| Host/sync overhead ratio p50 | 0.156202 % |
| Call throughput | 34.5833 calls/s |
| Output element throughput | 3.23401e+08 elements/s |
| Reference speedup | 0.972851 x |
| Reference GPU latency p50 | 28.1307 ms |
| Reference effective bandwidth p50 | 2.67164 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 37,749,636 bytes |
| Output writes | 37,405,456 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 75,155,092 bytes |
| Effective bandwidth, best | 2.70855 GB/s |
| Effective bandwidth, p50 | 2.59911 GB/s |
| Effective bandwidth, mean | 2.59553 GB/s |
| Effective bandwidth, p90 latency | 2.48694 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 75,498,496 bytes |
| Resident PyTorch reservation | 643,825,664 bytes |
| Incremental peak allocation | 37,748,736 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 113,247,232 bytes |
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
