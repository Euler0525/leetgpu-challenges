# Local practice report: challenges/easy/9_1d_convolution

- Backend: `cuda`
- Status: **passed**
- Updated: 2026-08-09T07:08:59+00:00

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
| functional_6 | passed | 4.76837e-07 | 1.08273e-07 |
| functional_7 | passed | 1.90735e-06 | 1.30357e-06 |
| functional_8 | passed | 5.72205e-06 | 0.000107066 |
| functional_9 | passed | 2.38419e-06 | 1.4275e-06 |
| functional_10 | passed | 4.76837e-07 | 1.45845e-06 |
| functional_11 | passed | 0 | 0 |
| functional_12 | passed | 2.98023e-08 | 1.02542e-07 |
| functional_13 | passed | 3.72529e-09 | 4.4125e-05 |

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 4.33254 ms |
| GPU latency p90 | 4.42491 ms |
| GPU latency p99 | 5.1266 ms |
| End-to-end latency p50 | 4.38045 ms |
| Host/sync overhead p50 | 0.0479062 ms |
| Host/sync overhead ratio p50 | 1.09364 % |
| Call throughput | 230.811 calls/s |
| Output element throughput | 3.45745e+08 elements/s |
| Reference speedup | 407.153 x |
| Reference GPU latency p50 | 1764.01 ms |
| Reference effective bandwidth p50 | 0.00680268 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 6,008,188 bytes |
| Output writes | 5,991,816 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 12,000,004 bytes |
| Effective bandwidth, best | 3.01554 GB/s |
| Effective bandwidth, p50 | 2.76974 GB/s |
| Effective bandwidth, mean | 2.75833 GB/s |
| Effective bandwidth, p90 latency | 2.71192 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0.000144958 |
| Performance max relative error | 0.186774 |
| Resident PyTorch allocation | 20,519,936 bytes |
| Resident PyTorch reservation | 102,760,448 bytes |
| Incremental peak allocation | 0 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 20,519,936 bytes |
| Peak PyTorch reservation | 102,760,448 bytes |

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
