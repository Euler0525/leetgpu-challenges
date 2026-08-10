# Local practice report: challenges/easy/54_swiglu

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T08:33:39+00:00

## Correctness

Passed 7/7 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 0 | 0 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.049152 ms |
| GPU latency p90 | 0.0691328 ms |
| GPU latency p99 | 0.0753955 ms |
| End-to-end latency p50 | 0.0662 ms |
| Host/sync overhead p50 | 0.017048 ms |
| Host/sync overhead ratio p50 | 25.7523 % |
| Call throughput | 20345.1 calls/s |
| Output element throughput | 1.01725e+09 elements/s |
| Reference speedup | 1.27995 x |
| Reference GPU latency p50 | 0.062912 ms |
| Reference effective bandwidth p50 | 9.53713 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 400,000 bytes |
| Output writes | 200,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 600,000 bytes |
| Effective bandwidth, best | 16.6223 GB/s |
| Effective bandwidth, p50 | 12.207 GB/s |
| Effective bandwidth, mean | 11.7836 GB/s |
| Effective bandwidth, p90 latency | 8.67895 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 600,576 bytes |
| Resident PyTorch reservation | 4,194,304 bytes |
| Incremental peak allocation | 400,384 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 1,000,960 bytes |
| Peak PyTorch reservation | 4,194,304 bytes |

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
