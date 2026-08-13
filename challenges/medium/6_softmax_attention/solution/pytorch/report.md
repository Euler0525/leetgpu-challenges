# Local practice report: challenges/medium/6_softmax_attention

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-12T11:51:57+00:00

## Correctness

Passed 5/5 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 0 | 0 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.074896 ms |
| GPU latency p90 | 0.102502 ms |
| GPU latency p99 | 0.179668 ms |
| End-to-end latency p50 | 0.08775 ms |
| Host/sync overhead p50 | 0.012854 ms |
| Host/sync overhead ratio p50 | 14.6484 % |
| Call throughput | 13351.8 calls/s |
| Output element throughput | 8.75027e+08 elements/s |
| Reference speedup | 0.900449 x |
| Reference GPU latency p50 | 0.06744 ms |
| Reference effective bandwidth p50 | 11.6612 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 524,288 bytes |
| Output writes | 262,144 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 786,432 bytes |
| Effective bandwidth, best | 12.1905 GB/s |
| Effective bandwidth, p50 | 10.5003 GB/s |
| Effective bandwidth, mean | 9.2833 GB/s |
| Effective bandwidth, p90 latency | 7.67233 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 9,306,112 bytes |
| Resident PyTorch reservation | 25,165,824 bytes |
| Incremental peak allocation | 1,310,720 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 10,616,832 bytes |
| Peak PyTorch reservation | 25,165,824 bytes |

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
