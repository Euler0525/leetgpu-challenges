# Local practice report: challenges/easy/1_vector_add

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-04T12:09:59+00:00

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
| GPU latency p50 | 1.48378 ms |
| GPU latency p90 | 1.51672 ms |
| GPU latency p99 | 1.58759 ms |
| End-to-end latency p50 | 1.5002 ms |
| Host/sync overhead p50 | 0.016424 ms |
| Host/sync overhead ratio p50 | 1.09479 % |
| Call throughput | 673.956 calls/s |
| Output element throughput | 1.68489e+10 elements/s |
| Reference speedup | 1.00961 x |
| Reference GPU latency p50 | 1.49803 ms |
| Reference effective bandwidth p50 | 200.263 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 100,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 300,000,000 bytes |
| Effective bandwidth, best | 229.24 GB/s |
| Effective bandwidth, p50 | 202.187 GB/s |
| Effective bandwidth, mean | 201.84 GB/s |
| Effective bandwidth, p90 latency | 197.795 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 301,989,888 bytes |
| Resident PyTorch reservation | 1,814,036,480 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 301,989,888 bytes |
| Peak PyTorch reservation | 1,814,036,480 bytes |

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
