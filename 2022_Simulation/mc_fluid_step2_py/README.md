# Phase 2：带自由滑移边界的蒙特卡洛流体模拟（`mc_fluid_step2_py`）

在 Phase 1 基础上前进到**含固体障碍物的 2D 无粘流动**，并施加自由滑移（壁面法向速度为零）边界条件。
本阶段给出两条实现路线——自实现的梯度 WoS 与外部 WoSX 库——并以网格流函数法作为确定性参考解做定量对比。

## 一、算法要点

- **流函数法**：求解 Poisson 方程 `∇²Ψ = ω`，边界条件 `Ψ = 0`（单连通域 Dirichlet）；
  速度由 `v = −∇×Ψ = (−∂Ψ/∂y, ∂Ψ/∂x)` 恢复。`Ψ = const` 自动保证壁面法向速度为零，无需压力投影。
- **路线 A：自实现梯度 WoS**（`core/wos.py`，Numba）
  - 单条 WoS 路径估计 Ψ：以到最近边界的距离 `R` 为半径在圆内均匀采样源项，累加 `πR²·ω·G(R,r)`，再跳到球面上继续；
  - **梯度 WoS** 估计 `∇Ψ`：第一球用**对侧采样（对偶采样）**处理球面边界项以降方差，再叠加球内源项梯度贡献；
  - `velocity_at(x) = (−∂Ψ/∂y, ∂Ψ/∂x)`。
- **路线 B：WoSX 库**（`core/wosx_simulator.py`、`core/wosx_stream_function.py`）
  - 障碍物写成 `.obj` 线段边界，交由 WoSX 的 `GeometricQueries` / `FcpwDirichletBoundaryHandler` / `WalkOnSpheres` 求解；
  - 每个时间步批量求解所有域内网格点的 Ψ，再由中心差分得到速度场；
  - 支持三种时间积分格式：`euler`（一阶）、`rk2`（中点法）、`mac`（预测-校正：用预测涡量重解速度场后再用平均速度回溯）。
- **参考解**：`core/grid_stream_solver.py` 用五点差分 + `scipy.sparse.spsolve` 解 `∇²Ψ = −ω`，
  固体内部与域外通过掩码强制 `Ψ = 0`。
- **无障碍退化**：无固体时直接使用 Phase 1 的 Biot–Savart 批量估计（`core/biot_savart.py`，共享采样点 + `prange`）。

## 二、目录结构

```
mc_fluid_step2_py/
├── main.py                    # 入口 1：自实现梯度 WoS（Numba）
├── wosx_main.py               # 入口 2：WoSX 求解器 + 时间格式选择
├── compare_mc_vs_grid.py      # 入口 3：WoSX vs 网格流函数法（含守恒量评估）
├── make_circle_obj.py         # 生成圆形边界 circle.obj
├── circle.obj                 # 示例边界（64 段折线圆）
├── core/
│   ├── types.py               # Vec2 / RNG
│   ├── grid.py                # Grid 涡量网格
│   ├── geometry.py            # Geometry：外边界 + 圆形/多边形障碍物，距离与最近点查询
│   ├── wos.py                 # WoS_solver（Numba 单路径 WoS + 梯度 WoS + 对偶采样）
│   ├── biot_savart.py         # Numba 批量 Biot–Savart（无障碍时使用）
│   ├── simulator.py           # 原始逐点 WoS 模拟器
│   ├── wosx_stream_function.py# WoSX 作为 Poisson 求解器的封装
│   ├── wosx_simulator.py      # WoSX 模拟器（euler / rk2 / mac）
│   ├── grid_stream_solver.py  # 网格流函数参考解（有限差分 + spsolve）
│   └── boundary.obj           # WoSX 使用的边界文件（按需自动生成）
├── render/visualizer.py       # GIF 动画 + 误差报告
└── output/                    # 运行产物
```

## 三、核心模块

| 文件 | 内容 |
|---|---|
| `core/types.py` | `Vec2`（含 `__neg__`、`__rmul__`、`__truediv__`）、`RNG`（`uniform` / `uniform_in_box`） |
| `core/grid.py` | `Grid(nx,ny,dx,ox,oy)`：涡量网格、双线性 `get_vort()`、`grid_pos()` |
| `core/geometry.py` | `Geometry`：`add_circle` / `add_rectangle`、`distance()`、`closest_point()`、`inside_domain()`（自由滑移边界与回溯点的域内约束都依赖它） |
| `core/wos.py` | `WoS_solver`：`_solve_psi_walk()`（`@njit` 单路径 WoS 估计 Ψ）、`estimate_psi_grad()`（梯度 WoS + 对偶采样）、`velocity_at()`；常量 `EPSILON=1e-4`、`MAX_STEPS=200` |
| `core/biot_savart.py` | `@njit` 核函数与 `_estimate_velocity_batch`：所有网格点共享同一批采样点 + `prange` 并行（避开并行 RNG 竞态） |
| `core/simulator.py` | 原版模拟器：有障碍走逐点 WoS，无障碍走批量 Biot–Savart + 半拉格朗日回溯 |
| `core/wosx_simulator.py` | `Simulator(nx,ny,dt,geometry,time_scheme)`：`_init_wosx()`、`_solve_psi()`、`_compute_velocity_from_psi()`、`_advect_back()`（euler/rk2/mac） |
| `core/wosx_stream_function.py` | `WoSXStreamFunctionSolver`：直接在查询点上求解 `∇²Ψ = ω, Ψ = 0`（`n_walks`、`epsilon`、`max_walk_length` 可调） |
| `core/grid_stream_solver.py` | `GridStreamFunctionSolver`：`solve_psi()`（五点差分）、`compute_velocity()`、`advect_omega()`、`step()` |
| `render/visualizer.py` | `save_animation()`：导出 GIF 并统计环量守恒 / 最大涡量变化 |

> ⚠️ `core/__init__.py` 末尾把 **WoSX 版 `Simulator`** 重新导出，因此 `from core import Simulator` 得到的是 WoSX 版本；
> 若要使用原版（逐点 WoS）模拟器，请显式 `from core.simulator import Simulator`。

## 四、脚本与运行参数

### 1) `main.py` —— 自实现梯度 WoS（Numba）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nx` / `--ny` | 64 / 64 | 网格分辨率 |
| `--dt` | 0.1 | 时间步长 |
| `--nmc` | 128 | 每个查询点的 WoS 路径数 |
| `--total_time` | 5.0 | 总模拟时长 |
| `--output_dir` | `output` | 输出目录 |
| `--fps` | 10 | 动画帧率 |
| `--obstacle_radius` | 0.3 | 圆形障碍物半径 |
| `--obstacle_center_x` / `--obstacle_center_y` | 0.0 / 0.0 | 障碍物圆心 |

```bash
python main.py --nx 64 --ny 64 --nmc 128 --obstacle_radius 0.2 --total_time 5
```

输出：`output/vorticity_with_obstacle.gif`、`output/last_frame.png`、`output/error_report.txt`。
实测（`output/error_report.txt`，128²、dt=0.2、nmc=2048、T=3.0）：总耗时 82.6 s（平均 5.5 s/步），
总环量 0.4482 → 0.4406（约 −1.7%），最大涡量 0.998 → 0.987。

### 2) `wosx_main.py` —— WoSX 求解器 + 时间格式选择

参数与 `main.py` 相同，额外提供：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--time_scheme` | `euler` | 时间积分格式：`euler`（一阶）/ `rk2`（中点法）/ `mac`（预测-校正） |

```bash
python wosx_main.py --nx 64 --nmc 64 --obstacle_radius 0.1 --time_scheme rk2 --total_time 2
```

输出：`output/wosx_vorticity_with_obstacle.gif`。首次运行会在 `core/` 下自动生成 `boundary.obj`。
**依赖 WoSX 库**（未安装请改用 `main.py`）。

### 3) `compare_mc_vs_grid.py` —— WoSX vs 网格流函数法

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nx` / `--ny` | 64 / 64 | 网格分辨率 |
| `--dt` | 0.05 | 时间步长 |
| `--total_time` | 1.0 | 总模拟时长 |
| `--nmc` | 64 | WoSX 每次求解的随机游走条数 |
| `--obstacle_radius` | 0.3 | 障碍物半径 |
| `--obstacle_center_x` / `--obstacle_center_y` | 0.0 / 0.0 | 障碍物圆心 |
| `--output_dir` | `output` | 输出目录 |
| `--time_scheme` | `euler` | `euler` / `rk2` / `mac` |

```bash
python compare_mc_vs_grid.py --nx 64 --ny 64 --dt 0.05 --nmc 64 --obstacle_radius 0.1 --total_time 3.0
```

输出：`output/step2_wosx_vs_grid_report_{time_scheme}.txt`、`output/step2_wosx_vs_grid_{time_scheme}.png`
（初始/末态涡量、两者差值 `|WoSX − Grid|`，以及环量 Γ、L1、Enstrophy、L2 范数变化率的柱状图）。
指标定义：`Γ = ∫ω dA`、`Z = ∫ω² dA`、`L1 = ∫|ω| dA`、`L2 = √Z`；误差为域内相对 L2 误差。

### 4) `make_circle_obj.py`

无命令行参数，按 `center=(0,0)`、`radius=0.3`、`n_segments=64` 生成 `circle.obj`
（顶点 `v x y 0` + 线段 `l i j`），供 WoSX 作为边界网格使用。

## 五、实测结果

### 5.1 三种时间积分格式（64²、dt=0.05、T=3.0 s、r=0.1；`output/step2_wosx_vs_grid_report_*.txt`）

| 格式 | WoSX L2 | WoSX 耗时 | 环量变化 | Enstrophy 变化 | Grid L2 / 耗时 |
|---|---|---|---|---|---|
| euler | 0.227166 | 234.94 s | −0.14% | −3.99% | 0.220934 / 5.85 s |
| rk2 | 0.226344 | 246.38 s | −0.22% | −4.05% | 0.220934 / 6.77 s |
| mac | 0.226632 | 400.84 s | −0.29% | −3.93% | 0.220934 / 5.68 s |

→ 三种格式的 WoSX 结果差异 < 0.4%，二阶格式没有带来精度收益，而 `mac` 因每步多解一次 Poisson 方程
使耗时增加约 70%，**并非性价比更高的选择**。

### 5.2 WoSX vs 网格流函数法

| 算例 | 方法 | L2 误差 | 耗时 | 内存 |
|---|---|---|---|---|
| 64²、T=3.0、r=0.1（euler） | WoSX | 0.2272 | 234.94 s | 112.95 MB |
| | Grid | 0.2209 | 5.85 s | 120.06 MB |
| 32²、T=2.0、r=0.2（旧版 MC-WoS） | MC (WoS) | 0.2163 | 202.16 s | 166.78 MB |
| | Grid | 0.2116 | 1.49 s | 168.83 MB |

**结论**

- WoSX 在**守恒性**上明显占优：环量变化 −0.14% vs 网格法 −3.42%，Enstrophy −3.99% vs −10.72%
  （无网格耗散、边界处理无几何离散误差）；
- 代价是**两个数量级的耗时差**（约 40 倍），且 L2 精度与网格法相当，因此工程上应优先选网格法，
  只有在网格生成困难/内存受限时才考虑 WoS 类方法；
- 两者在障碍物附近的差异最大：WoS 的误差来自统计噪声与曲率处理，网格法的误差来自几何离散。

## 六、注意事项

- **`core/wos.py` 中障碍物被硬编码为圆心 (0,0)、半径 0.3**（`obstacle_data = [0,0,0.3]`），
  因此 `main.py` 的 `--obstacle_radius` / `--obstacle_center_*` 对该求解器**不完全生效**；
  需要任意障碍物时请使用 `wosx_main.py`（它会依命令行参数重新生成 `boundary.obj`）。
- **`boundary.obj` 缓存**：`_init_wosx()` 只在文件不存在时生成边界，修改障碍物参数后若结果不变，
  请先删除 `core/boundary.obj`。
- **回溯点越界处理**：半拉格朗日回溯点若落到固体内部或域外，会被 `Geometry.closest_point()` 投影回边界
  并向域内推 1e-4，以避免涡量泄漏；`dt` 过大（相对网格上的速度）会放大该修正带来的误差。
- **性能**：每个时间步都要求解一次流函数，WoS/WoSX 的成本随 `nmc`、网格点数近似线性增长；
  128²、nmc=2048 时约 5.5 s/步（`output/error_report.txt`）。
- `output/step2_wosx_vs_grid_report.txt`、`step2_mc_vs_grid_report.txt` 为早期版本脚本产物（不含 `time_scheme` 字段），
  保留用于对照（128²、r=0.3 时 WoSX L2 0.2539 vs Grid 0.4383）。

