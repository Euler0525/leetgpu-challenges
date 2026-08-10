# Local practice report: challenges/easy/68_sigmoid

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T09:04:49+00:00

## Correctness

Passed 14/14 cases.

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
| functional_12 | passed | 0 | 0 |
| functional_13 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 4.0448 ms |
| GPU latency p90 | 4.06169 ms |
| GPU latency p99 | 4.07458 ms |
| End-to-end latency p50 | 4.0614 ms |
| Host/sync overhead p50 | 0.0166002 ms |
| Host/sync overhead ratio p50 | 0.408731 % |
| Call throughput | 247.231 calls/s |
| Output element throughput | 1.23616e+10 elements/s |
| Reference speedup | 0.500103 x |
| Reference GPU latency p50 | 2.02282 ms |
| Reference effective bandwidth p50 | 197.744 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 200,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000,000 bytes |
| Effective bandwidth, best | 99.3957 GB/s |
| Effective bandwidth, p50 | 98.8924 GB/s |
| Effective bandwidth, mean | 98.8292 GB/s |
| Effective bandwidth, p90 latency | 98.4813 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 400,000,000 bytes |
| Resident PyTorch reservation | 3,414,163,456 bytes |
| Incremental peak allocation | 200,000,000 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 600,000,000 bytes |
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
