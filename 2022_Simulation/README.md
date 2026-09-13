# 蒙特卡洛流体模拟方法复现与误差评估（2022）

复现 Rioux-Lavoie、Sugimoto 等人（*A Monte Carlo Method for Fluid Simulation*, ACM TOG 41(6), 2022）
提出的蒙特卡洛流体求解器：用纯 Python 从 2D 无粘欧拉方程逐步扩展到 3D Navier–Stokes 方程，
并与解析解（Taylor–Green 涡旋）及确定性方法（网格流函数法、涡量粒子法）做定量误差评估。

- 作者：潘硕（PB24020526）
- 课程报告：[`Report/src/2022_report.tex`](Report/src/2022_report.tex)（xelatex 编译），配图见 [`Report/pic/`](Report/pic)

## 一、核心思想

以**涡量 ω** 为原始变量，避开压力的全局投影求解：

1. **速度重建**：Biot–Savart 定律的蒙特卡洛积分
   2D `u(x) = (1/2π)∫ ω(y)(x−y)⊥/|x−y|² dy`；3D `u(x) = (1/4π)∫ ω(y)×(x−y)/|x−y|³ dy`。
2. **时间推进**：半拉格朗日后向追踪平流涡量。
3. **网格缓存**：蒙特卡洛样本取自上一时间步的涡量场，从而截断递归积分（避免指数复杂度）。
4. **随机表示**：粘性用 Feynman–Kac 高斯扰动（Wiener 过程）表示；3D 涡拉伸用涡量分段法估计。
5. **工程优化**：NumPy 向量化 + Numba（`njit`/`prange`）JIT；3D 中叠加重要性采样与控制变量（方差缩减）。

## 二、目录结构

```
2022_Simulation/
├── README.md
├── requirements.txt
├── mc_fluid_step1_py/                  # Phase 1：2D 无粘欧拉方程
│   ├── main.py                         # 双高斯涡旋演化动画（Numba 加速）
│   ├── compare_2dtg.py                 # 2D Taylor–Green 解析解验证（含 Biot–Savart 速度误差）
│   ├── study_nmc.py                    # 误差 ~ 样本数 n 收敛性研究
│   ├── compare_with_particles.py       # MC vs 涡量粒子法（误差-时间曲线 + Pareto 曲线）
│   ├── compare_vortex_pair.py          # 与 2024 论文场景 3 同初始场（涡对）的复算
│   ├── save_frames.py                  # 导出涡对算例关键帧
│   ├── gpu_fluid_2022.py               # CuPy GPU 版 Biot–Savart（可选依赖）
│   ├── core/                           # types / grid / biot_savart / simulator / vortex_solver
│   ├── render/visualizer.py            # GIF 动画 + 环量/极值误差报告
│   └── output/                         # 运行产物（gif、误差曲线、报告 txt）
├── mc_fluid_step2_py/                  # Phase 2：2D 自由滑移边界（流函数 + WoS/WoSX）
│   ├── main.py                         # 自实现梯度 WoS（Numba）求解流函数
│   ├── wosx_main.py                    # WoSX 求解器，支持 --time_scheme euler/rk2/mac
│   ├── compare_mc_vs_grid.py           # WoSX vs 网格流函数法（含守恒量评估）
│   ├── make_circle_obj.py              # 生成圆形边界 circle.obj
│   ├── circle.obj, core/boundary.obj   # 边界几何（WoSX 输入）
│   ├── core/                           # types / grid / geometry / wos / biot_savart / simulator
│   │                                   # wosx_stream_function / wosx_simulator / grid_stream_solver
│   ├── render/visualizer.py
│   └── output/
├── mc_fluid_step3_py/                  # Phase 3：3D Navier–Stokes（Feynman–Kac + 涡拉伸）
│   ├── main.py                         # 3D 涡量切片动画
│   ├── compare_tg3d.py                 # 3D Taylor–Green（Re=1600）基准与参考数据对比
│   ├── ref.txt                         # OpenSBLI 64³ 参考数据（Ek、Z）
│   ├── core/                           # types3d / grid3d / biot_savart3d / simulator3d / grid_solve3d
│   ├── render/visualizer3d.py
│   └── output/
├── Report/                             # 课程报告
│   ├── src/                            # 2022_report.tex / .pdf / 汇报 PDF
│   └── pic/step1|step2|step3/          # 报告插图
└── test/                               # WoSX 环境自检脚本（test_wosx.py、test2.py）
```

## 三、安装

开发环境：Python 3.13（建议 3.9+）。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy / matplotlib / pillow / numba / scipy / psutil

# 可选：GPU 版 Biot–Savart
pip install cupy-cuda12x                 # 按本机 CUDA 版本选择

# 可选：WoSX 边界求解库（Phase 2 的 wosx_main.py / compare_mc_vs_grid.py 需要）
# 官方未发布到 PyPI，需从上游源码编译安装；安装后用 test/test_wosx.py 自检
python test/test_wosx.py
```

## 四、快速开始

```bash
# ---------- Phase 1：2D 无粘 ----------
cd mc_fluid_step1_py
python main.py --nx 64 --ny 64 --dt 0.1 --nmc 256 --total_time 10      # → graph/vorticity_evolution.gif
python compare_2dtg.py --nx 32 --dt 0.05 --nmc 2048 --total_time 1.0   # → output/tg2d_*.png, tg2d_report.txt
python study_nmc.py --nx 64 --dt 0.02 --total_time 1.0 --repeats 3     # → output/nmc_study.txt 等

# ---------- Phase 2：带障碍物的自由滑移边界 ----------
cd ../mc_fluid_step2_py
python main.py --nx 64 --ny 64 --nmc 128 --obstacle_radius 0.3 --total_time 5         # 自实现 WoS（Numba）
python wosx_main.py --nmc 64 --obstacle_radius 0.1 --time_scheme rk2 --total_time 2   # WoSX
python compare_mc_vs_grid.py --nx 64 --nmc 64 --total_time 3 --obstacle_radius 0.1

# ---------- Phase 3：3D Navier–Stokes ----------
cd ../mc_fluid_step3_py
python main.py --nx 16 --ny 16 --nz 16 --nmc 8 --nd 4 --total_time 1
python compare_tg3d.py --nx 32 --nmc 512 --dt 0.02 --total_time 2.0
python compare_tg3d.py --nx 64 --nmc 1024 --dt 0.05 --total_time 1.0 --control_variate
```

> 注意：蒙特卡洛积分成本为 `O(网格点数 × 样本数)`，完整复现实验耗时较长
> （Phase 2 的 WoSX 对比约数分钟～数十分钟，Phase 3 的 64³ 算例单步即数秒以上），
> 上例参数已按可承受成本调整；各脚本的默认参数见对应子项目 README。

## 五、三个阶段

| Phase | 目录 | 方程 | 维度 | 边界 | 粘性 | 拉伸 | 方差缩减 | 关键算法 |
|-------|------|------|------|------|------|------|----------|----------|
| 1 | `mc_fluid_step1_py` | Euler | 2D | 无（开放/周期域） | – | – | – | 均匀采样 Biot–Savart + 半拉格朗日回溯 + 双缓冲网格缓存（Numba） |
| 2 | `mc_fluid_step2_py` | Euler | 2D | 自由滑移（Ψ=0） | – | – | 梯度 WoS 对偶采样 / WoSX 控制变量 | 流函数 Poisson（∇²Ψ=ω）+ WoS/WoSX，时间格式 euler/rk2/mac |
| 3 | `mc_fluid_step3_py` | Navier–Stokes | 3D | 无（开放/周期域） | ✅ | ✅ | 重要性采样 + 控制变量 | Feynman–Kac 扩散 + 涡量分段拉伸 + 3D Biot–Savart 重建 |

## 六、主要结论（详见报告与各阶段 README）

- **Phase 1**：MC 速度重建的统计误差随样本数下降，但很快触底——`nmc` 从 128 增到 512 时平均 L2 误差仅由
  0.0899 降到 0.0846（<6%），log–log 斜率偏离理论值 `O(1/√n)`，说明**插值/时间离散的系统误差主导**。
  与涡量粒子法相比，同等精度下 MC 慢约 2 个数量级（64²、dt=0.01、T=1.0：MC nmc=1024 耗时 1503 s、
  L2=0.0812；涡粒子 46 s、L2=0.0821）。
- **Phase 2**：WoSX 无网格求解的守恒性明显优于网格差分（64²、T=3.0 s：环量变化 −0.14% vs −3.42%，
  Enstrophy −3.99% vs −10.72%），但耗时约为网格法的 40 倍；`rk2`/`mac` 高阶时间格式并未带来精度提升。
- **Phase 3**：64³、Re=1600 的 Taylor–Green 算例中，初始拟涡能 Z 与参考解接近（0.375 vs 0.374），
  但 Z 的增长速率远慢于参考解（t=1 时仅 0.4969），定量偏差很大——数值耗散与统计噪声仍是 3D 场景的主要瓶颈。

## 七、已知限制

- **计算效率**：远低于确定性网格法，3D + 边界场景尤甚。
- **数值耗散**：线性（三线性）插值 + 半拉格朗日格式会抹平高频涡量细节，误差随时间近似线性增长。
- **统计噪声**：样本不足时速度估计出现非物理尖峰（代码中用速度/涡量限幅兜底）。
- **WoSX 依赖**：未安装时 Phase 2 的 `wosx_main.py`、`compare_mc_vs_grid.py` 无法运行，
  可退回 `main.py`（自实现 Numba WoS）；`core/wos.py` 中障碍物参数硬编码为圆心 (0,0)、半径 0.3。
- **控制变量尚未充分验证**：Phase 3 的 `--control_variate` 路径建议与无 CV 结果对照使用。

## 八、参考文献

1. Damien Rioux-Lavoie, Ryusuke Sugimoto, et al. *A Monte Carlo Method for Fluid Simulation*. ACM TOG 41(6), 2022.
2. Ryusuke Sugimoto, Christopher Batty, Toshiya Hachisuka. *Velocity-Based Monte Carlo Fluids*. SIGGRAPH 2024.
3. Rohan Sawhney, Keenan Crane. *Monte Carlo Geometry Processing*. ACM TOG 39(4), 2020.
4. J. DeBonis. *Solutions of the Taylor-Green Vortex Problem Using High-Resolution Explicit Finite Difference Methods*. AIAA 2013（`mc_fluid_step3_py/ref.txt` 参考数据来源，由 OpenSBLI 整理）。

## 九、许可

本项目仅用于课程学习与学术研究。代码为作者基于论文公开思想自行实现的复现版本，不含原作者受版权保护的代码；
`mc_fluid_step3_py/ref.txt` 参考数据版权归 OpenSBLI 作者所有。
