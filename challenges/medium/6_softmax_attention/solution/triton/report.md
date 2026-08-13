# Local practice report: challenges/medium/6_softmax_attention

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-12T14:08:18+00:00

## Correctness

Passed 5/5 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 4.76837e-07 | 1.11181e-07 |
| functional_1 | passed | 4.76837e-07 | 1.11181e-07 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 3.33067e-16 | 2.17785e-06 |
| functional_4 | passed | 2.79397e-09 | 6.55049e-06 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.217088 ms |
| GPU latency p90 | 0.232237 ms |
| GPU latency p99 | 0.561422 ms |
| End-to-end latency p50 | 0.24915 ms |
| Host/sync overhead p50 | 0.032062 ms |
| Host/sync overhead ratio p50 | 12.8686 % |
| Call throughput | 4606.43 calls/s |
| Output element throughput | 3.01887e+08 elements/s |
| Reference speedup | 0.731206 x |
| Reference GPU latency p50 | 0.158736 ms |
| Reference effective bandwidth p50 | 4.95434 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 524,288 bytes |
| Output writes | 262,144 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 786,432 bytes |
| Effective bandwidth, best | 4.43931 GB/s |
| Effective bandwidth, p50 | 3.62264 GB/s |
| Effective bandwidth, mean | 3.42049 GB/s |
| Effective bandwidth, p90 latency | 3.38634 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 3.76895e-09 |
| Performance max relative error | 0.062364 |
| Resident PyTorch allocation | 9,306,112 bytes |
| Resident PyTorch reservation | 25,165,824 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 9,306,112 bytes |
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
