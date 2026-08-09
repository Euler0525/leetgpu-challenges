# Local practice report: challenges/easy/9_1d_convolution

- Backend: `pytorch`
- Status: **failed**
- Updated: 2026-08-09T06:20:54+00:00

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

## Errors

- `run`: Tensor-likes are not close!

Mismatched elements: 1123258 / 1497954 (75.0%)
Greatest absolute difference: 0.019745349884033203 at index (150748,) (up to 0.0001 allowed)
Greatest relative difference: 165.125 at index (1278817,) (up to 0.0001 allowed)

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu124
- Triton: 3.1.0
- CUDA runtime: 12.4
- Python: 3.12.3
- Platform: Windows-11-10.0.26200-SP0
