# Local practice report: challenges/easy/54_swiglu

- Backend: `triton`
- Status: **passed**
- Updated: 2026-08-10T08:38:32+00:00

## Correctness

Passed 7/7 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 4.76837e-07 | 1.08709e-07 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 0 | 0 |
| functional_5 | passed | 3.05176e-05 | 3.7607e-06 |
| functional_6 | passed | 3.05176e-05 | 3.07933e-06 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.024576 ms |
| GPU latency p90 | 0.030496 ms |
| GPU latency p99 | 0.096135 ms |
| End-to-end latency p50 | 0.0371 ms |
| Host/sync overhead p50 | 0.012524 ms |
| Host/sync overhead ratio p50 | 33.7574 % |
| Call throughput | 40690.1 calls/s |
| Output element throughput | 2.03451e+09 elements/s |
| Reference speedup | 2.27083 x |
| Reference GPU latency p50 | 0.055808 ms |
| Reference effective bandwidth p50 | 10.7511 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 400,000 bytes |
| Output writes | 200,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 600,000 bytes |
| Effective bandwidth, best | 30.8388 GB/s |
| Effective bandwidth, p50 | 24.4141 GB/s |
| Effective bandwidth, mean | 21.3343 GB/s |
| Effective bandwidth, p90 latency | 19.6747 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0.00012207 |
| Performance max relative error | 4.12921e-06 |
| Resident PyTorch allocation | 600,576 bytes |
| Resident PyTorch reservation | 4,194,304 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 600,576 bytes |
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
