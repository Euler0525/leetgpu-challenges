# Matrix Multiplication — Triton 测试用例

运行以下命令时，本地练习程序会执行本文件列出的测试：

```powershell
python .\scripts\local_practice.py run easy/2_matrix_multiplication --backend triton
```

所有张量均为 CUDA 上的 `torch.float32`，并验证 `C = A @ B`。正确性阶段共 15 个用例：1 个示例用例和 14 个功能用例；全部通过后，才会执行 1 个性能用例。

## 正确性用例

| 本地报告名称 | 源名称 | `M × N × K` | `A`、`B` 的生成方式 |
|---|---|---:|---|
| `example` | example | 2 × 2 × 2 | 固定值，见下文 |
| `functional_1` | basic_2x2 | 2 × 2 × 2 | 固定值，见下文 |
| `functional_2` | basic_1x3_3x1 | 1 × 3 × 1 | 固定值，见下文 |
| `functional_3` | identity_matrix | 3 × 3 × 3 | 固定值，见下文 |
| `functional_4` | zero_matrix | 2 × 2 × 2 | 固定值，见下文 |
| `functional_5` | rectangular_matrices | 2 × 3 × 1 | 固定值，见下文 |
| `functional_6` | small_square | 4 × 4 × 4 | `uniform(-10.0, 10.0)` |
| `functional_7` | medium_rectangular | 8 × 6 × 10 | `uniform(-10.0, 10.0)` |
| `functional_8` | large_rectangular | 16 × 12 × 20 | `uniform(-10.0, 10.0)` |
| `functional_9` | tall_matrix | 32 × 8 × 16 | `uniform(-10.0, 10.0)` |
| `functional_10` | wide_matrix | 8 × 16 × 32 | `uniform(-10.0, 10.0)` |
| `functional_11` | single_element | 1 × 1 × 1 | `uniform(-1.0, 1.0)` |
| `functional_12` | single_row | 1 × 5 × 3 | `uniform(-1.0, 1.0)` |
| `functional_13` | single_column | 5 × 3 × 1 | `uniform(-1.0, 1.0)` |
| `functional_14` | max_dimensions | 8,192 × 6,144 × 4,096 | `uniform(-1.0, 1.0)` |

随机用例会在每次命令运行时重新生成；因此上表完整定义了它们的形状和取值分布，但不固定每个元素的数值。

### 固定值用例

#### `example` 与 `functional_1`: 2 × 2 × 2

```text
A = [[1.0, 2.0],
     [3.0, 4.0]]
B = [[5.0, 6.0],
     [7.0, 8.0]]
C = [[19.0, 22.0],
     [43.0, 50.0]]
```

#### `functional_2`: 1 × 3 × 1

```text
A = [[1.0, 2.0, 3.0]]
B = [[4.0],
     [5.0],
     [6.0]]
C = [[32.0]]
```

#### `functional_3`: 3 × 3 × 3

```text
A = [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]]
B = [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]]
C = [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]]
```

#### `functional_4`: 2 × 2 × 2

```text
A = [[0.0, 0.0], [0.0, 0.0]]
B = [[0.0, 0.0], [0.0, 0.0]]
C = [[0.0, 0.0], [0.0, 0.0]]
```

#### `functional_5`: 2 × 3 × 1

```text
A = [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
B = [[1.0],
     [2.0],
     [3.0]]
C = [[14.0],
     [32.0]]
```

## 性能用例

| 名称 | `M × N × K` | `A`、`B` 的生成方式 | 输出 |
|---|---:|---|---|
| performance | 8,192 × 6,144 × 4,096 | `RandTensor(shape, -10.0, 10.0)` | `C` 为 8,192 × 4,096 的输出张量 |

性能用例同样会先与 PyTorch 参考实现比对；验证通过后，运行 5 次预热及 30 次计时迭代。
