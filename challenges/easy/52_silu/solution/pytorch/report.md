# Local practice report: challenges/easy/52_silu

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T08:16:42+00:00

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
| GPU latency p50 | 0.025616 ms |
| GPU latency p90 | 0.0322624 ms |
| GPU latency p99 | 0.0636739 ms |
| End-to-end latency p50 | 0.0413 ms |
| Host/sync overhead p50 | 0.015684 ms |
| Host/sync overhead ratio p50 | 37.9758 % |
| Call throughput | 39038.1 calls/s |
| Output element throughput | 1.95191e+09 elements/s |
| Reference speedup | 1.42973 x |
| Reference GPU latency p50 | 0.036624 ms |
| Reference effective bandwidth p50 | 10.9218 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000 bytes |
| Output writes | 200,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000 bytes |
| Effective bandwidth, best | 20.5592 GB/s |
| Effective bandwidth, p50 | 15.6152 GB/s |
| Effective bandwidth, mean | 14.1879 GB/s |
| Effective bandwidth, p90 latency | 12.3983 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 400,384 bytes |
| Resident PyTorch reservation | 4,194,304 bytes |
| Incremental peak allocation | 200,192 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 600,576 bytes |
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
