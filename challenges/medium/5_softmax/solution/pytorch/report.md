# Local practice report: challenges/medium/5_softmax

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-09-08T08:21:54+00:00

## Correctness

Passed 11/11 cases.

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
| functional_9 | passed | 4.65661e-10 | 1.4177e-07 |
| functional_10 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.17384 ms |
| GPU latency p90 | 0.177139 ms |
| GPU latency p99 | 0.179191 ms |
| End-to-end latency p50 | 0.203 ms |
| Host/sync overhead p50 | 0.02916 ms |
| Host/sync overhead ratio p50 | 14.3645 % |
| Call throughput | 5752.42 calls/s |
| Output element throughput | 2.87621e+09 elements/s |
| Reference speedup | 0.588955 x |
| Reference GPU latency p50 | 0.102384 ms |
| Reference effective bandwidth p50 | 39.0686 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 2,000,000 bytes |
| Output writes | 2,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 4,000,000 bytes |
| Effective bandwidth, best | 23.2515 GB/s |
| Effective bandwidth, p50 | 23.0097 GB/s |
| Effective bandwidth, mean | 22.9612 GB/s |
| Effective bandwidth, p90 latency | 22.5811 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 1.45519e-11 |
| Performance max relative error | 4.1537e-07 |
| Resident PyTorch allocation | 4,000,768 bytes |
| Resident PyTorch reservation | 44,040,192 bytes |
| Incremental peak allocation | 2,000,384 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 6,001,152 bytes |
| Peak PyTorch reservation | 44,040,192 bytes |

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
