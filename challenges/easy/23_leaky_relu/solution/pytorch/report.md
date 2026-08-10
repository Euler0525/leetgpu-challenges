# Local practice report: challenges/easy/23_leaky_relu

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T06:59:45+00:00

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
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 7.33838 ms |
| GPU latency p90 | 7.37953 ms |
| GPU latency p99 | 7.4309 ms |
| End-to-end latency p50 | 7.36455 ms |
| Host/sync overhead p50 | 0.0261661 ms |
| Host/sync overhead ratio p50 | 0.355298 % |
| Call throughput | 136.27 calls/s |
| Output element throughput | 6.81349e+09 elements/s |
| Reference speedup | 1.14925 x |
| Reference GPU latency p50 | 8.43366 ms |
| Reference effective bandwidth p50 | 47.429 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 200,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000,000 bytes |
| Effective bandwidth, best | 54.7708 GB/s |
| Effective bandwidth, p50 | 54.5079 GB/s |
| Effective bandwidth, mean | 54.4589 GB/s |
| Effective bandwidth, p90 latency | 54.204 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 400,000,000 bytes |
| Resident PyTorch reservation | 3,414,163,456 bytes |
| Incremental peak allocation | 450,000,384 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 850,000,384 bytes |
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
