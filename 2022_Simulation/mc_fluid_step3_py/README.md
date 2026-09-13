# Phase 3：3D Navier–Stokes 蒙特卡洛流体模拟（`mc_fluid_step3_py`）

把求解器从 2D 无粘扩展到 **3D 不可压缩 Navier–Stokes 方程**的涡量形式：

```
∂ω/∂t + (u·∇)ω = (ω·∇)u + ν∇²ω,      u = Biot–Savart(ω)
```

即同时处理**平流**、**粘性扩散**与**涡拉伸**三项，并用 3D Taylor–Green（Re=1600）标准算例与外部参考数据做定量比较。
本阶段同时引入了本项目的主要方差缩减手段（重要性采样 + 控制变量）。

## 一、算法要点

- **平流（advection）**：半拉格朗日后向追踪 `x_b = x − u(x)Δt`，速度由 3D Biot–Savart 的蒙特卡洛估计给出。
- **粘性（Feynman–Kac）**：在回溯点上叠加 Wiener 过程的高斯扰动 `√(2νΔt)·ξ`，
  用 `nd` 条子路径的样本均值近似条件期望 `w(x) = E[w(x + √(2νΔt)ξ)]`。
- **涡拉伸（vorticity stretching）**：涡量分段法——沿涡量方向在 `x_b ± (h/2)·ω̂` 处插值速度并做差，
  得到 `(ω·∇)u ≈ (u(x+) − u(x−))/h` 对 ω 的作用，`h = dx`，再做显式欧拉更新。
- **速度重建（3D Biot–Savart）**：`u(x) = (1/4π)∫ ω(y)×(x−y)/|x−y|³ dy`，包含三项优化：
  - **重要性采样**：按涡量幅值构建 CDF（`build_vorticity_cdf` / `importance_sample`）抽取样本点，并用 `pdf` 加权，
    这是默认开启的方差缩减手段；
  - **周期镜像**：`periodic=True` 时对样本点做 3×3×3 镜像求和，保证周期域的正确性；
  - **控制变量（可选）**：只估计涡量增量 `Δω` 对应的 `Δu`，再叠加前一步的速度 `u_prev`；
    首步用 `initial_nmc`（默认 `4 × nmc`）提高采样数。
- **数值兜底**：速度限幅 `|u| ≤ 2.0`、涡量幅值限幅 `|ω| ≤ 5.0`、单个拉伸项限幅 `10.0`（硬编码于 `simulator3d.py`）。
- **确定性参考实现**：`core/grid_solve3d.py` 的 `GridSolver3D`（半拉格朗日平流 + 显式扩散 + `spsolve` 压力投影），
  用于与 MC 结果对照。

## 二、目录结构

```
mc_fluid_step3_py/
├── main.py                 # 入口 1：3D 双高斯涡旋的 z 中面切片动画
├── compare_tg3d.py         # 入口 2：3D Taylor–Green（Re=1600）基准与参考数据对比
├── ref.txt                 # OpenSBLI 64³ 参考数据（t、∫Ek、∫Z，源自 DeBonis 2013）
├── core/
│   ├── types3d.py          # Vec3 / RNG3D
│   ├── grid3d.py           # Grid3D：涡量/速度场 + 三线性插值
│   ├── biot_savart3d.py    # 3D Biot–Savart（重要性采样 / 均匀采样 / 周期镜像 / 控制变量基础）
│   ├── simulator3d.py      # Simulator3D：速度重建 + 平流/扩散/拉伸
│   └── grid_solve3d.py     # 确定性参考解（有限差分 + 压力投影）
├── render/visualizer3d.py  # 切片 GIF 动画 + 误差报告
└── output/                 # 运行产物
```

## 三、核心模块

| 文件 | 内容 |
|---|---|
| `core/types3d.py` | `Vec3`（`+ − *`、`dot/norm/cross`）、`RNG3D`（`uniform` / `gaussian`，Box–Muller 带缓存） |
| `core/grid3d.py` | `Grid3D(nx,ny,nz,dx,ox,oy,oz,periodic)`：`vort`/`vel` 形状 `(nz,ny,nx,3)`；`@njit` `trilinear_interp_numba`；向量化批量插值 `get_vort_interp_batch` / `get_vel_interp_batch`；`slice_z0` |
| `core/biot_savart3d.py` | `@njit(parallel=True)` `_numba_biot_savart_batch`（含 3×3×3 周期镜像与 `pdf` 加权）；`build_vorticity_cdf` / `importance_sample`（按涡量幅值采样）；`uniform_sample`；`estimate_velocity_batch(use_importance, periodic, Lx/Ly/Lz)`；`estimate_velocity`（逐点，供扩展） |
| `core/simulator3d.py` | `@njit(parallel=True)` `_numba_update_vorticity`（平流 + Feynman–Kac 扩散 + 涡量分段拉伸 + 限幅）；`Simulator3D(nx,ny,nz,dt,nu=0,L=1,use_control_variate=False,initial_nmc=None,use_periodic=False)`，含 `nmc`、`nd`、`h`、控制变量缓存 `prev_vort`/`prev_vel` |
| `core/grid_solve3d.py` | `GridSolver3D`：`advect()`、`diffuse()`、`project()`（`scipy.sparse.spsolve` 解压力 Poisson）、`get_vorticity()`、`step()`；另含解析 TG 涡量 `tg_vorticity()` |
| `render/visualizer3d.py` | `save_animation_3d()`：逐帧记录 z 中面涡量幅值、导出 GIF，并输出切片涡量积分/最大值的误差报告 |

## 四、脚本与运行参数

### 1) `main.py` —— 3D 涡量切片动画

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nx` / `--ny` / `--nz` | 16 / 16 / 16 | 网格分辨率 |
| `--dt` | 0.1 | 时间步长 |
| `--nmc` | 8 | 速度重建的 MC 样本数 |
| `--nu` | 0.0 | 运动粘性系数 |
| `--nd` | 4 | Feynman–Kac 扩散子步数 |
| `--total_time` | 1.0 | 总模拟时长 |
| `--output_dir` | `output` | 输出目录 |
| `--fps` | 5 | 动画帧率 |

```bash
python main.py --nx 32 --ny 32 --nz 32 --dt 0.1 --nu 0.0 --nmc 32 --nd 4 --total_time 1.0
```

输出：`output/vorticity_3d_slice.gif`（z 中面涡量幅值动画）、`output/last_frame.png`、`output/error_report.txt`。
实测量级（`output/error_report.txt`，32³、nmc=32、nd=4、T=1.0）：总耗时 147.2 s（约 14.7 s/步），
切片涡量积分 28.12 → 26.04，最大涡量 0.814 → 1.066。

### 2) `compare_tg3d.py` —— 3D Taylor–Green（Re=1600）基准

初始涡量场（`k = π/L`）：

```
ωx = −k·cos(kx)sin(ky)sin(kz),  ωy = −k·sin(kx)cos(ky)sin(kz),  ωz = 2k·sin(kx)sin(ky)cos(kz)
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nx` | 64 | 网格分辨率（nx=ny=nz） |
| `--dt` | 0.01 | 时间步长 |
| `--total_time` | 20.0 | 总模拟时长（无量纲时间） |
| `--nmc` | 1024 | 速度重建 MC 样本数 |
| `--nu` | 1/1600 | 粘性系数（Re=1600） |
| `--L` | π | 半边长（计算域 [−L,L]³） |
| `--output_dir` | `output` | 输出目录 |
| `--ref_file` | `ref.txt` | 参考数据文件（OpenSBLI 64³） |
| `--control_variate` | 关 | 开启控制变量方差缩减 |
| `--initial_nmc` | `4 × nmc` | 控制变量首步的全速度采样数 |

```bash
python compare_tg3d.py --nx 32 --nmc 512 --dt 0.02 --total_time 2.0
python compare_tg3d.py --nx 64 --nmc 1024 --dt 0.05 --total_time 1.0 --control_variate
```

输出：`tg3d_report.txt`（逐时刻 `Ek`、`Z`，并在有参考数据时追加峰值误差）、
`tg3d_kinetic_energy.png`、`tg3d_enstrophy.png`；若 `ref_file` 存在，再输出
`tg3d_Ek_comparison.png`、`tg3d_Z_comparison.png`。指标定义（体积平均）：

```
Ek = 0.5 · Σ|u|²·dx³ / V,     Z = 0.5 · Σ|ω|²·dx³ / V
```

`ref.txt` 为 OpenSBLI 64³ 的三列数据：无量纲时间、积分动能、积分拟涡能（见文件头部注释，源自 DeBonis 2013）。

## 五、实测结果

64³、`nmc=1024`、`ν=1/1600`、`dt=0.05`、`T=1.0`（`output/tg3d_report.txt` 与 `tg3d_Z_comparison.png`）：

| t | 0.05 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| 模拟 Ek | 0.1562 | 0.1562 | 0.1566 | 0.1581 | 0.1609 |
| 模拟 Z | 0.3753 | 0.3904 | 0.4191 | 0.4547 | 0.4969 |
| 参考 Ek | 0.1250 | 0.1248 | 0.1195 | 0.1145 | 0.1123 |
| 参考 Z | 0.3744 | 0.3786 | 2.4038 | 3.6079 | 4.0861 |

> 参考列对应 `ref.txt` 中最接近的采样时刻，仅作趋势对比。

**结论**

- **初始拟涡能与参考解吻合**（0.3753 vs 0.3744），说明涡量场初始化与短时积分是可信的；
- 但 **Z 的增长速率远慢于参考解**（T=1 时仅 0.497，而参考在 t≈0.5 已达 2.40），
  峰值统计给出「峰值拟涡能相对误差 92.1%、峰值时刻相对误差 89.1%」——需要注意该对比的 `T=1` 远未覆盖参考峰值（t≈9.14），
  因此报告中的峰值误差反映的是**拟涡能演化整体偏慢**这一系统偏差，而非完全等价于峰值精度；
- 模拟动能与参考解存在约 25% 的偏差且在 T=1 内几乎不衰减，说明 MC 速度重建的统计噪声/离散误差
  对动能的影响不可忽略；**MC 格式的数值耗散与统计噪声仍是 3D 场景的主要瓶颈**。

## 六、注意事项

- **计算成本**：N = 64³ 时每步需 `N × nmc ≈ 2.7×10⁸` 次核函数求值；实测 32³、nmc=32、nd=4 约 14.7 s/步
  （`output/error_report.txt`），64³ 单步为分钟量级，因此 Re=1600 的完整算例（t≈9–20）需要长时间预算/集群，
  仓库中保存的是短时对比结果。
- **硬编码限幅**：速度 `≤ 2.0`、涡量幅值 `≤ 5.0`、拉伸项 `≤ 10.0`（`core/simulator3d.py`）。
  它们能抑制统计噪声引起的发散，但也会限制真实峰值，调试时需注意。
- **控制变量路径尚未系统验证**：`--control_variate` 会使用 `Δω → Δu + u_prev` 的估计，
  建议与关闭 CV 的结果对照使用；首步默认 `4 × nmc` 采样可通过 `--initial_nmc` 调整。
- **边界处理**：`compare_tg3d.py` 固定 `use_periodic=True`（TG 是周期流，配合 3×3×3 镜像求和）；
  非周期模式（`use_periodic=False`）下回溯点与扩散点会被 clamp 到「离边界 dx」以内。
- `output/` 中的 `tg3d_evolution.gif`、`tg3d_slice_compare.png`、`tg3d_dissipation_rate.png`
  为早期版本脚本的历史产物，当前 `compare_tg3d.py` 只生成上文列出的 4 张 PNG 与报告：
  截图对比图见 `Report/pic/step3/tg3d_Z_comparison.png`、`tg3d_Ek_comparison.png`。

