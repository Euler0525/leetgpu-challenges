# Local practice report: challenges/easy/1_vector_add

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-03T12:53:09+00:00

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
| GPU latency p50 | 1.27274 ms |
| GPU latency p90 | 1.28429 ms |
| GPU latency p99 | 1.30496 ms |
| End-to-end latency p50 | 1.28315 ms |
| Host/sync overhead p50 | 0.010414 ms |
| Host/sync overhead ratio p50 | 0.811595 % |
| Call throughput | 785.709 calls/s |
| Output element throughput | 1.96427e+10 elements/s |
| Reference speedup | 1.14488 x |
| Reference GPU latency p50 | 1.45714 ms |
| Reference effective bandwidth p50 | 205.883 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 100,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 300,000,000 bytes |
| Effective bandwidth, best | 237.414 GB/s |
| Effective bandwidth, p50 | 235.713 GB/s |
| Effective bandwidth, mean | 235.114 GB/s |
| Effective bandwidth, p90 latency | 233.592 GB/s |
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
