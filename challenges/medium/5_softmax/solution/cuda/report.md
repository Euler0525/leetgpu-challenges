# Local practice report: challenges/medium/5_softmax

- Backend: `cuda`
- Status: **failed**
- Updated: 2026-08-26T12:17:09+00:00

## Correctness

Passed 11/11 cases.

| Case | Status | Max absolute error | Max relative error |
|---|---:|---:|---:|
| example | passed | 0 | 0 |
| functional_1 | passed | 0 | 0 |
| functional_2 | passed | 0 | 0 |
| functional_3 | passed | 0 | 0 |
| functional_4 | passed | 4.65661e-10 | 7.8992e-08 |
| functional_5 | passed | 0 | 0 |
| functional_6 | passed | 0 | 0 |
| functional_7 | passed | 0 | 0 |
| functional_8 | passed | 0 | 0 |
| functional_9 | passed | 4.65661e-10 | 1.59216e-07 |
| functional_10 | passed | 0 | 0 |

## Errors

- `run`: Tensor-likes are not close!

Mismatched elements: 35759 / 500000 (7.2%)
Greatest absolute difference: 0.00495483074337244 at index (3833,) (up to 1e-05 allowed)
Greatest relative difference: 124.47174072265625 at index (93,) (up to 1e-05 allowed)

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.13.0+cu130
- Triton: 3.7.1
- CUDA runtime: 13.0
- Python: 3.14.6
- Platform: Windows-11-10.0.26200-SP0
