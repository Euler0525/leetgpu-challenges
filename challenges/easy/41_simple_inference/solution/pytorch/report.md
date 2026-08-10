# Local practice report: challenges/easy/41_simple_inference

- Backend: `pytorch`
- Status: **passed**
- Updated: 2026-08-10T08:06:18+00:00

## Correctness

Passed 9/9 cases.

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

## Benchmark

| Metric | Value |
|---|---:|
| GPU latency p50 | 0.104224 ms |
| GPU latency p90 | 0.109261 ms |
| GPU latency p99 | 0.119347 ms |
| End-to-end latency p50 | 0.1163 ms |
| Host/sync overhead p50 | 0.012076 ms |
| Host/sync overhead ratio p50 | 10.3835 % |
| Call throughput | 9594.72 calls/s |
| Output element throughput | 2.45625e+09 elements/s |
| Reference speedup | 1.00583 x |
| Reference GPU latency p50 | 0.104832 ms |
| Reference effective bandwidth p50 | 34.315 GB/s |

### Memory traffic and effective bandwidth

| Metric | Value |
|---|---:|
| Input reads | 2,573,312 bytes |
| Output writes | 1,024,000 bytes |
| In-place reads | 0 bytes |
| In-place writes | 0 bytes |
| Minimum total traffic | 3,597,312 bytes |
| Effective bandwidth, best | 35.8469 GB/s |
| Effective bandwidth, p50 | 34.5152 GB/s |
| Effective bandwidth, mean | 34.0531 GB/s |
| Effective bandwidth, p90 latency | 32.9241 GB/s |
| Arithmetic throughput | n/a (challenge has no generic FLOP count) |

### Performance-case validation and memory

| Metric | Value |
|---|---:|
| Performance output status | passed |
| Performance max absolute error | 0 |
| Performance max relative error | 0 |
| Resident PyTorch allocation | 12,116,992 bytes |
| Resident PyTorch reservation | 29,360,128 bytes |
| Incremental peak allocation | 2,072,576 bytes |
| Incremental peak reservation | 0 bytes |
| Peak PyTorch allocation | 14,189,568 bytes |
| Peak PyTorch reservation | 29,360,128 bytes |

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
