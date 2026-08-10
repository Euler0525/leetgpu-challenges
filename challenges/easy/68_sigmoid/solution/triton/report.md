# Local practice report: challenges/easy/68_sigmoid

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-10T09:07:26+00:00

## Correctness

Passed 14/14 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 5.96046e-08 | 8.1532e-08 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 5.96046e-08 | 8.1532e-08 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 5.96046e-08 | 8.1532e-08 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 0 | 0 |
| functional_7 | passed | 3.63798e-12 | 8.01355e-08 |
| functional_8 | passed | 3.72529e-09 | 7.85497e-08 |
| functional_9 | passed | 5.96046e-08 | 2.15711e-07 |
| functional_10 | passed | 1.19209e-07 | 2.22511e-07 |
| functional_11 | passed | 1.19209e-07 | 5.18325e-07 |
| functional_12 | passed | 1.19209e-07 | 1.77006e-07 |
| functional_13 | passed | 1.78814e-07 | 3.31617e-07 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 1.78258 ms |
| GPU latency p90 | 1.80634 ms |
| GPU latency p99 | 1.80838 ms |
| End-to-end latency p50 | 1.79505 ms |
| Host/sync overhead p50 | 0.012474 ms |
| Host/sync overhead ratio p50 | 0.694913 % |
| Call throughput | 560.986 calls/s |
| Output element throughput | 2.80493e+10 elements/s |
| Reference speedup | 1.13451 x |
| Reference GPU latency p50 | 2.02235 ms |
| Reference effective bandwidth p50 | 197.79 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 200,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000,000 bytes |
| Effective bandwidth, best | 226.068 GB/s |
| Effective bandwidth, p50 | 224.394 GB/s |
| Effective bandwidth, mean | 224.027 GB/s |
| Effective bandwidth, p90 latency | 221.443 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 1.78814e-07 |
| Performance max relative error | 6.03324e-07 |
| Resident PyTorch allocation | 400,000,000 bytes |
| Resident PyTorch reservation | 3,414,163,456 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 400,000,000 bytes |
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
