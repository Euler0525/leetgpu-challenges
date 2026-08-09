# Local practice report: challenges/easy/7_color_inversion

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-09T04:41:34+00:00

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
| GPU latency p50 | 0.73216 ms |
| GPU latency p90 | 0.73735 ms |
| GPU latency p99 | 0.775954 ms |
| End-to-end latency p50 | 0.7447 ms |
| Host/sync overhead p50 | 0.01254 ms |
| Host/sync overhead ratio p50 | 1.6839 % |
| Call throughput | 1365.82 calls/s |
| Output element throughput | 1.14573e+11 elements/s |
| Reference speedup | 2.48728 x |
| Reference GPU latency p50 | 1.82109 ms |
| Reference effective bandwidth p50 | 92.1274 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 0 bytes |
| Output writes | 0 bytes |
| In-place reads | 83,886,080 bytes |
| In-place writes | 83,886,080 bytes |
| Minimum total traffic | 167,772,160 bytes |
| Effective bandwidth, best | 230.761 GB/s |
| Effective bandwidth, p50 | 229.147 GB/s |
| Effective bandwidth, mean | 228.348 GB/s |
| Effective bandwidth, p90 latency | 227.534 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 167,772,160 bytes |
| Resident PyTorch reservation | 2,602,565,632 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 167,772,160 bytes |
| Peak PyTorch reservation | 2,602,565,632 bytes |

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
