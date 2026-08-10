# Local practice report: challenges/easy/52_silu

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-10T08:21:11+00:00

## Correctness

Passed 8/8 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 2.38419e-07 | 8.34296e-08 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 1.49012e-08 | 1.04733e-07 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 4.76837e-07 | 2.28365e-07 |
| functional_7 | passed | 9.53674e-07 | 1.18945e-07 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.017264 ms |
| GPU latency p90 | 0.0225536 ms |
| GPU latency p99 | 0.0378902 ms |
| End-to-end latency p50 | 0.02645 ms |
| Host/sync overhead p50 | 0.009186 ms |
| Host/sync overhead ratio p50 | 34.7297 % |
| Call throughput | 57924 calls/s |
| Output element throughput | 2.8962e+09 elements/s |
| Reference speedup | 2.30584 x |
| Reference GPU latency p50 | 0.039808 ms |
| Reference effective bandwidth p50 | 10.0482 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 200,000 bytes |
| Output writes | 200,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 400,000 bytes |
| Effective bandwidth, best | 28.0269 GB/s |
| Effective bandwidth, p50 | 23.1696 GB/s |
| Effective bandwidth, mean | 20.838 GB/s |
| Effective bandwidth, p90 latency | 17.7355 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 9.53674e-07 |
| Performance max relative error | 1.19154e-07 |
| Resident PyTorch allocation | 400,384 bytes |
| Resident PyTorch reservation | 4,194,304 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 400,384 bytes |
| Peak PyTorch reservation | 4,194,304 bytes |

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
