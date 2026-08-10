# Local practice report: challenges/easy/31_matrix_copy

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T07:27:27+00:00

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
| GPU latency p50 | 0.595904 ms |
| GPU latency p90 | 0.598016 ms |
| GPU latency p99 | 0.606128 ms |
| End-to-end latency p50 | 0.60625 ms |
| Host/sync overhead p50 | 0.010346 ms |
| Host/sync overhead ratio p50 | 1.70656 % |
| Call throughput | 1678.12 calls/s |
| Output element throughput | 2.81542e+10 elements/s |
| Reference speedup | 1.04301 x |
| Reference GPU latency p50 | 0.621536 ms |
| Reference effective bandwidth p50 | 215.945 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 67,108,864 bytes |
| Output writes | 67,108,864 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 134,217,728 bytes |
| Effective bandwidth, best | 226.768 GB/s |
| Effective bandwidth, p50 | 225.234 GB/s |
| Effective bandwidth, mean | 225.358 GB/s |
| Effective bandwidth, p90 latency | 224.438 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 134,217,728 bytes |
| Resident PyTorch reservation | 1,142,947,840 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 134,217,728 bytes |
| Peak PyTorch reservation | 1,142,947,840 bytes |

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
