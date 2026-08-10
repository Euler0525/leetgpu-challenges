# Local practice report: challenges/easy/63_interleave

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-10T09:18:32+00:00

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
| functional_9 | passed | 0 | 0 |
| functional_10 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 2.0172 ms |
| GPU latency p90 | 2.03366 ms |
| GPU latency p99 | 2.04923 ms |
| End-to-end latency p50 | 2.0316 ms |
| Host/sync overhead p50 | 0.0144 ms |
| Host/sync overhead ratio p50 | 0.708801 % |
| Call throughput | 495.737 calls/s |
| Output element throughput | 2.47868e+10 elements/s |
| Reference speedup | 1.55582 x |
| Reference GPU latency p50 | 3.1384 ms |
| Reference effective bandwidth p50 | 127.453 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000,000 bytes |
| Output writes | 200,000,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000,000 bytes |
| Effective bandwidth, best | 199.604 GB/s |
| Effective bandwidth, p50 | 198.295 GB/s |
| Effective bandwidth, mean | 198.107 GB/s |
| Effective bandwidth, p90 latency | 196.689 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 401,326,592 bytes |
| Resident PyTorch reservation | 3,414,163,456 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 401,326,592 bytes |
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
