# Local practice report: challenges/easy/65_geglu

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-10T11:54:22+00:00

## Correctness

Passed 7/7 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 2.98023e-07 | 0.111111 |
| functional_4 | passed | 0 | 0 |
| functional_5 | passed | 3.05176e-05 | 0.00170944 |
| functional_6 | passed | 3.05176e-05 | 2.79028e+302 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.02048 ms |
| GPU latency p90 | 0.0245344 ms |
| GPU latency p99 | 0.0296928 ms |
| End-to-end latency p50 | 0.03075 ms |
| Host/sync overhead p50 | 0.01027 ms |
| Host/sync overhead ratio p50 | 33.3984 % |
| Call throughput | 48828.1 calls/s |
| Output element throughput | 2.44141e+10 elements/s |
| Reference speedup | 5.175 x |
| Reference GPU latency p50 | 0.105984 ms |
| Reference effective bandwidth p50 | 56.6123 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 4,000,000 bytes |
| Output writes | 2,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 6,000,000 bytes |
| Effective bandwidth, best | 308.388 GB/s |
| Effective bandwidth, p50 | 292.969 GB/s |
| Effective bandwidth, mean | 274.256 GB/s |
| Effective bandwidth, p90 latency | 244.555 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 6.10352e-05 |
| Performance max relative error | 7.30928e+302 |
| Resident PyTorch allocation | 6,000,640 bytes |
| Resident PyTorch reservation | 44,040,192 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 6,000,640 bytes |
| Peak PyTorch reservation | 44,040,192 bytes |

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
