# Phase 1：2D 无粘蒙特卡洛流体模拟（`mc_fluid_step1_py`）

复现 2022 论文的基础原型：求解二维不可压缩无粘欧拉方程

```
∂ω/∂t + (u·∇)ω = 0,     u = Biot–Savart(ω)
```

核心流程为「半拉格朗日后向追踪 + 蒙特卡洛 Biot–Savart 速度重建 + 双缓冲网格缓存」，
所有热点函数使用 Numba（`njit` / `prange`）加速，并允许通过 `nu > 0` 打开 Feynman–Kac 形式的粘性扩散。

## 一、算法要点

- **涡量–速度公式**：以 ω 为原始变量，用 Biot–Savart 定律由 ω 重建 u，无需压力投影的全局求解。
- **均匀蒙特卡洛采样**：在计算域内均匀撒点，以样本均值估计速度积分（2D 核 `ω(y)(x−y)⊥/(2π|x−y|²)`）。
- **半拉格朗日回溯**：`x_b = x − u(x)Δt`，再对上一时刻涡量场做双线性插值得到新涡量。
- **网格缓存（caching）**：只保留相邻两个时间步的涡量场（`grid[0]` / `grid[1]`），
  使单步成本为 `O(N · n_mc)`，而不是递归蒙特卡洛积分的指数增长——这是 2022 论文的关键工程点。
- **可选扩散**：`nu > 0` 时在回溯点上叠加 `√(2νΔt)·ξ` 的高斯扰动，`nd` 条子路径取期望。
- **周期边界**：`periodic=True` 时 Biot–Savart 核会对 3×3 周期镜像求和，消除边界截断误差。

## 二、目录结构

```
mc_fluid_step1_py/
├── main.py                    # 入口 1：双高斯涡旋演化动画
├── compare_2dtg.py            # 入口 2：2D Taylor–Green 解析解验证
├── study_nmc.py               # 入口 3：误差 ~ 样本数收敛性研究
├── compare_with_particles.py  # 入口 4：MC vs 涡量粒子法对比
├── compare_vortex_pair.py     # 入口 5：复算 2024 论文场景 3（涡对）
├── save_frames.py             # 导出涡对算例关键帧
├── gpu_fluid_2022.py          # CuPy GPU 版 Biot–Savart（可选）
├── core/
│   ├── types.py               # Vec2 / RNG
│   ├── grid.py                # Grid（涡量网格 + 双线性插值）
│   ├── biot_savart.py         # Numba Biot–Savart（单点 / 批量）
│   ├── simulator.py           # Simulator（平流 + 扩散主循环）
│   └── vortex_solver.py       # 涡量粒子法（对比基准）
├── render/visualizer.py       # GIF 动画 + 误差报告
└── output/                    # 运行产物
```

## 三、核心模块

| 文件 | 内容 |
|---|---|
| `core/types.py` | `Vec2`（`+ - *`、`dot/norm2/norm`、`perp`）、`RNG`（Box–Muller 高斯，带缓存） |
| `core/grid.py` | `Grid(nx,ny,dx,ox,oy,periodic)`：`get_vort()` 双线性插值、`set_vort()`、`grid_pos()`、`area` |
| `core/biot_savart.py` | `@njit` `interp_vort_numba`；`@njit(parallel=True)` `biot_savart_velocity_numba`（批量求速度、固定 seed、支持 3×3 周期镜像）；`BiotSavart.kernel / estimate_velocity` |
| `core/simulator.py` | `@njit(parallel=True)` `update_vorticity_numba`（半拉格朗日平流 + 扩散）；`Simulator(nx, ny, dt, nu=0, L=1.0, periodic=False, nmc=256, nd=4)`，默认生成两个高斯涡旋，`set_vorticity_field(func)` 可自定义初始场 |
| `core/vortex_solver.py` | `VortexParticleSolver`：涡量粒子法（中点法 RK2 推进 + 高斯核重投影到网格），作为 MC 的对比基准 |
| `render/visualizer.py` | `save_animation()`：按帧采样并导出 GIF，同时统计环量守恒与最大涡量衰减，写出 `error_report.txt` |

## 四、脚本与运行参数

### 1) `main.py` —— 双高斯涡旋演化

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nx` / `--ny` | 64 / 64 | 网格分辨率 |
| `--dt` | 0.1 | 时间步长 |
| `--nmc` | 256 | 每次速度查询的 MC 样本数 |
| `--total_time` | 10.0 | 总模拟时长 |
| `--output_dir` | `graph` | 输出目录 |
| `--fps` | 20 | 动画帧率 |

```bash
python main.py --nx 64 --ny 64 --dt 0.1 --nmc 128 --total_time 5
```

输出：`graph/vorticity_evolution.gif`、`graph/last_frame.png`、`graph/error_report.txt`
（逐帧记录总环量与最大涡量，并打印模拟总耗时、平均单步耗时）。

### 2) `compare_2dtg.py` —— 2D Taylor–Green 严格验证

参考解（周期域，粘性衰减）：`ω = 2k·sin(kx)sin(ky)·e^{−2νk²t}`，`u = sin(kx)cos(ky)`，`v = −cos(kx)sin(ky)`。
脚本先做静态验证（Biot–Savart 重建的瞬时速度 vs 解析速度），再做动态验证（涡量 L2 误差随时间演化）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nx` | 32 | 网格分辨率 |
| `--dt` | 0.005 | 时间步长 |
| `--total_time` | 0.5 | 总模拟时长 |
| `--nu` | 0.05 | 粘性系数（解析衰减率） |
| `--L` | π | 半边长（计算域 [−L,L]²） |
| `--nmc` | 2000 | MC 样本数 |
| `--nd` | 8 | 扩散子步数 |
| `--output_dir` | `output` | 输出目录 |

```bash
python compare_2dtg.py --nx 64 --dt 0.02 --nmc 4096 --total_time 1.0 --nu 0.0
```

输出：`tg2d_velocity_magnitude.png`、`tg2d_velocity_quiver.png`（速度幅值/矢量对比）、
`tg2d_error.png`（L2 误差时间曲线，semilog）、`tg2d_slice.png`（末态涡量全场对比）、`tg2d_report.txt`。

典型结果（`output/tg2d_report.txt`，32²、dt=0.05、ν=0、nmc=2048）：Biot–Savart 速度 L2 相对误差 ≈ 0.20
（主要由网格离散而非采样数决定），涡量 L2 误差在 t=1.0 时约 0.077。

### 3) `study_nmc.py` —— 误差随样本数 n 的收敛性

流程：固定 64²、dt=0.02、T=1.0，遍历 `nmc_list`，每组重复 `repeats` 次取均值/标准差，绘制 log–log 曲线并叠加 `O(1/√n)` 参考线。

| 参数 | 默认值 |
|---|---|
| `--nx` / `--ny` | 64 / 64 |
| `--dt` | 0.02 |
| `--total_time` | 1.0 |
| `--nmc_list` | `16,32,64,128,256,512` |
| `--repeats` | 5 |
| `--output_dir` | `output` |

```bash
python study_nmc.py --nx 64 --dt 0.02 --nmc_list "8,32,64,128,256" --repeats 3
```

输出 `output/nmc_study.txt`、`output/nmc_error_convergence.png`。实测：

| nmc | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| 平均 L2 误差 | 0.1250 | 0.1125 | 0.0992 | 0.0899 | 0.0865 | 0.0846 |
| 标准差 | 0.0035 | 0.0078 | 0.0068 | 0.0009 | 0.0007 | 0.0005 |

结论：统计误差随 n 下降但很快触底（128 → 512 误差仅降约 6%，而成本 ×4），log–log 斜率偏离 −0.5，
说明**系统误差（插值 + 时间离散）主导**；`n ≈ 128–256` 已是性价比拐点。

### 4) `compare_with_particles.py` —— MC vs 涡量粒子法

用 Taylor–Green 参考场比较两种方法，记录 L2 相对误差、计算时间与进程内存（`psutil`）。

| 参数 | 默认值 |
|---|---|
| `--nx` / `--ny` | 32 / 32 |
| `--dt` | 0.02 |
| `--total_time` | 0.5 |
| `--nmc_list` | `16,64,256,1024` |
| `--output_dir` | `output` |

```bash
python compare_with_particles.py --nx 64 --ny 64 --dt 0.01 --total_time 1.0
```

输出：`output/comparison_report.txt`、`output/error_vs_time.png`、`output/pareto_curve.png`。实测（64²、dt=0.01、T=1.0）：

| 方法 | L2 误差 | 计算时间 (s) |
|---|---|---|
| MC (nmc=16) | 0.1146 | 23.1 |
| MC (nmc=64) | 0.0906 | 87.6 |
| MC (nmc=256) | 0.0834 | 388.5 |
| MC (nmc=1024) | 0.0812 | 1503.4 |
| Vortex Particle | 0.0821 | 46.3 |

### 5) `compare_vortex_pair.py` 与 `save_frames.py`

两者均**无命令行参数**：使用 2024 论文场景 3 的涡对初始场（中心 (−0.7, ±1/6)、半径 0.8/6，64²、dt=0.05、
nmc=128、40 步）复算 2022 版算法，分别输出 `compare_output/2022_vortex_pair_frames.png`
（5 个时刻的涡量快照拼图）与 `compare_output/frames/frame_*.png`（单帧图片），用于两篇论文的交叉对比。

### 6) `gpu_fluid_2022.py`（可选，GPU）

CuPy 实现：`@cp.fuse` 融合双线性插值 + 分块（`chunk=8192`）的 Biot–Savart 积分，把最耗时的速度重建搬到 GPU。

```bash
python gpu_fluid_2022.py --nx 64 --nmc 128 --steps 40          # 涡对初始场 + 计时
python gpu_fluid_2022.py --nx 64 --nmc 128 --steps 40 --taylor_green
```

需要 `cupy`（无 GPU/未安装 cupy 时请忽略此脚本）。

## 五、运行产物（`output/`）

| 文件 | 说明 |
|---|---|
| `vorticity.gif` | 涡量演化动画（历史版本 `main.py` 产物；当前默认输出到 `graph/`） |
| `error_report.txt` / `error_curve.png` / `error_vs_time.png` / `last_frame.png` | 误差统计与曲线（部分为早期脚本产物） |
| `nmc_study.txt` / `nmc_error_convergence.png` | 样本数收敛性研究 |
| `comparison_report.txt` / `pareto_curve.png` | MC vs 涡量粒子法对比 |
| `tg2d_report.txt` / `tg2d_error.png` / `tg2d_slice.png` | Taylor–Green 验证（另两张速度对比图已归入 `Report/pic/step1/`） |

## 六、注意事项

- **必须安装 numba**：`core/biot_savart.py`、`core/simulator.py` 在导入时即使用 `@njit`，缺少 numba 会直接 ImportError。
- **随机数**：Numba 内核使用固定 seed（12345 / 67890，`compare_2dtg.py` 用 42）保证可复现；
  批量估计让所有网格点共享同一批样本以提高缓存命中率，代价是样本间存在相关性，误差估计会略偏乐观。
- **默认输出目录**：`main.py` 默认 `graph/`，对比类脚本默认 `output/`；README 引用的图片位于 `output/`。
- `output/error_report.txt` 为早期脚本（如已移除的 `compare_tg.py`）留下的产物，重跑当前脚本会覆盖同名文件。
- 非周期域下 Biot–Savart 积分域被直接截断，边界附近积分缺失使误差分布高度不均（实测 `L∞ ≫ L2`）。

