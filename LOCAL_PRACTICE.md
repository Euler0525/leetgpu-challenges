# 本地 GPU 练习

本仓库为每道题提供互不覆盖的 PyTorch、Triton 和 CUDA 练习目录，并使用统一命令完成正确性测试、性能测试和 Profiler 采样。原有题目、starter 和线上提交脚本不会被修改。

## 初始化

当前题目已经初始化。以后从上游同步了新题，可再次运行：

```powershell
python scripts/local_practice.py init
```

初始化是幂等的：只补充缺失文件，不覆盖你写过的解答或已有报告。单独初始化一道题：

```powershell
python scripts/local_practice.py init easy/1_vector_add
```

每个受支持的后端目录包含：

```text
solution/<backend>/
├── solution.py 或 solution.cu
├── report.json
├── report.md
├── profile_trace.json
├── solution.nsys-rep（CUDA，可选）
└── solution.ncu-rep（CUDA，可选）
```

`41_simple_inference` 由原题限定为仅支持 PyTorch；运行器会跳过它的 Triton 和 CUDA 后端。

## 终端运行

只运行示例和功能正确性测试：

```powershell
python scripts/local_practice.py test easy/1_vector_add --backend pytorch
```

只运行性能用例和 Profiler：

```powershell
python scripts/local_practice.py benchmark medium/22_gemm --backend triton
```

依次运行正确性、性能和 Profiler：

```powershell
python scripts/local_practice.py run easy/1_vector_add --backend cuda
```

CUDA 后端可额外生成能被 Nsight 图形界面完整打开的原生报告：

```powershell
python scripts/local_practice.py benchmark easy/52_silu --backend cuda --nsys --ncu
```

Nsys 与 NCU 都只采集性能用例在预热后的一次 `solve` 调用，不包含参考实现、正确性测试、NVCC 编译或重复 benchmark。NCU 默认使用 `detailed` 指标集，也可选择 `basic`、`full` 或 `roofline`：

```powershell
python scripts/local_practice.py benchmark easy/52_silu --backend cuda --ncu --ncu-set roofline
```

使用变体题解时，原生报告沿用题解文件名。例如下面的命令生成 `solution4.nsys-rep` 和 `solution4.ncu-rep`：

```powershell
python scripts/local_practice.py run medium/4_reduction --backend cuda --solution challenges/medium/4_reduction/solution/cuda/solution4.cu --nsys --ncu
```

`--backend all` 只会为其中的 CUDA 后端生成原生报告。显式选择 PyTorch 或 Triton 时不能传入 Nsight 参数。`--no-profile` 只关闭 PyTorch Chrome trace，不影响 `--nsys` 或 `--ncu`。

同一道题依次测试全部受支持后端：

```powershell
python scripts/local_practice.py run easy/1_vector_add --backend all
```

也可在题目目录内运行：

```powershell
python ../../../scripts/local_practice.py run . --backend pytorch
```

默认预热 5 次、测量 30 次。可按需调整：

```powershell
python scripts/local_practice.py run easy/1_vector_add --backend triton --warmup 10 --repeat 100
```

命令成功返回退出码 `0`，测试或编译失败返回非零退出码，因此可以直接用于终端脚本。完整参数可运行：

```powershell
python scripts/local_practice.py --help
python scripts/local_practice.py run --help
```

---

Compute Sanitizer 对代码的功能正确性执行不同类型的检查：

- memcheck：内存访问错误和泄漏检测；

- racecheck：共享内存数据访问危险检测工具；

- initcheck：未初始化设备全局内存访问检测工具；

- synccheck：线程同步危险检测；


```powershell
compute-sanitizer --tool memcheck python .\scripts\local_practice.py test easy/2_matrix_multiplication --backend cuda
```

## 报告内容

每次运行会更新对应后端目录下的固定报告文件，文件已纳入 Git，可直接查看改动：

- 功能用例通过数，以及每个输出的最大绝对误差和相对误差
- CUDA Event 测得的 GPU 延迟 min、mean、p50、p90、p99、max 和标准差
- 包含同步开销的端到端延迟、主机/同步开销、调用吞吐和输出元素吞吐
- 输入读取、输出写入、原地读写的字节数，以及 best、p50、mean、p90 有效内存带宽
- 性能用例输出误差、相对题目 PyTorch 参考实现的加速比与参考实现带宽
- PyTorch 分配器常驻量、总峰值和增量峰值，以及完整软硬件环境
- PyTorch Profiler Chrome trace
- Nsys/NCU 工具版本、指标集、文件大小、题解 SHA-256、源码嵌入状态和最近一次生成错误

同一份解答通过 SHA-256 识别：单独运行 `test` 会保留已有性能结果和 trace，单独运行 `benchmark` 会保留已有正确性结果。解答文件一旦变化，旧性能结果不会被复用，避免报告与代码版本不一致。正确性、性能和 trace 分别记录自己的更新时间。

报告格式为 schema v3。原生报告先生成到 `out/local_practice/`，验证非空后再替换题解旁的固定文件；失败不会覆盖上一次成功文件。普通 benchmark 成功而某个 Nsight 工具失败时，报告状态为 `partial`，命令返回非零，并保留正确性与性能数据。题解哈希变化后，旧原生报告标记为 `stale`，但不会自动删除。

有效内存带宽按题目参数的最低读写字节数计算，不等同于硬件计数器采集的实际 DRAM 流量。仓库题目没有统一 FLOP 数元数据，因此报告不会给出容易误导的 TFLOPS；需要实际 DRAM 流量或算术吞吐时，可在安装 Nsight Compute 后结合 trace 深入分析。

`profile_trace.json` 可用 Perfetto 或兼容 Chrome trace 的工具打开。原生 CUDA 内核能否展开到逐 kernel 细节取决于本机 PyTorch 的 CUPTI 支持；trace 始终包含 `leetgpu::<backend>::solve` 区间，GPU 延迟始终由 CUDA Event 单独测量。

在题解目录中用图形界面打开原生报告：

```powershell
nsys-ui solution.nsys-rep
ncu-ui solution.ncu-rep
```

运行器不会自动启动 GUI。NCU 需要 NVIDIA GPU Performance Counters 权限；若出现 `ERR_NVGPUCTRPERM`，请在 NVIDIA Control Panel 的 Desktop/Developer → Manage GPU Performance Counters 中允许所有用户访问，或使用有权限的管理员会话。

CUDA 动态库和 Triton 编译缓存写入现有 Git 忽略目录 `out/local_practice/`。`*.nsys-rep` 与 `*.ncu-rep` 也被全局忽略，避免把大型二进制报告提交到 Git；可阅读的 `report.json`、`report.md`、题解源码和 Chrome trace 仍会被 Git 追踪。
