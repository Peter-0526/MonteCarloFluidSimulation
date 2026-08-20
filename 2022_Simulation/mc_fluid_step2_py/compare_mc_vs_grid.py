# compare_mc_vs_grid.py
import sys, os, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 使用 WoSX 模拟器
from core.wosx_simulator import Simulator
from core.geometry import Geometry
from core.types import Vec2
from core.grid_stream_solver import GridStreamFunctionSolver

# ============================================================
# 初始条件：两个高斯涡旋（避开障碍物）
# ============================================================
def init_vorticity(nx, ny, dx, ox, oy, geometry):
    grid = np.zeros((ny, nx))
    sigma = 0.2
    centers = [Vec2(-0.5, 0.0), Vec2(0.5, 0.0)]
    for j in range(ny):
        for i in range(nx):
            p = Vec2(ox + (i+0.5)*dx, oy + (j+0.5)*dx)
            if not geometry.inside_domain(p):
                continue
            w = 0.0
            for c in centers:
                d = (p - c).norm()
                w += np.exp(-d*d/(2*sigma*sigma))
            grid[j, i] = w
    return grid

# ============================================================
# 新增：计算全域积分量
# ============================================================
def circulation(omega, dx):
    """总环量 Γ = ∫∫ ω dA"""
    return np.sum(omega) * dx * dx

def enstrophy(omega, dx):
    """Enstrophy Z = ∫∫ ω² dA"""
    return np.sum(omega**2) * dx * dx

def l2_norm(omega, dx):
    """L2 范数 ||ω||₂ = sqrt(∫∫ ω² dA)"""
    return np.sqrt(enstrophy(omega, dx))

def l1_norm(omega, dx):
    """L1 范数 ∫∫ |ω| dA"""
    return np.sum(np.abs(omega)) * dx * dx

# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Step2: WoSX vs Grid Stream Function (with conservation diagnostics)')
    parser.add_argument('--nx', type=int, default=64, help='网格横向分辨率')
    parser.add_argument('--ny', type=int, default=64, help='网格纵向分辨率')
    parser.add_argument('--dt', type=float, default=0.05, help='时间步长')
    parser.add_argument('--total_time', type=float, default=1.0, help='总模拟时长')
    parser.add_argument('--nmc', type=int, default=64, help='WoSX 路径数')
    parser.add_argument('--obstacle_radius', type=float, default=0.3, help='障碍物半径')
    parser.add_argument('--obstacle_center_x', type=float, default=0.0)
    parser.add_argument('--obstacle_center_y', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default='output')
    # 新增：时间积分方案
    parser.add_argument('--time_scheme', type=str, default='euler',
                        choices=['euler', 'rk2', 'mac'],
                        help='Time integration scheme for WoSX: euler, rk2, mac')
    args = parser.parse_args()

    nx, ny = args.nx, args.ny
    dt, total_time = args.dt, args.total_time
    ox, oy = -1.0, -1.0
    dx = 2.0 / nx
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    geo = Geometry(outer_bounds=(-1.0, 1.0, -1.0, 1.0))
    geo.add_circle(Vec2(args.obstacle_center_x, args.obstacle_center_y), args.obstacle_radius)

    omega0 = init_vorticity(nx, ny, dx, ox, oy, geo)
    area_element = dx * dx

    # ---------- WoSX方法 ----------
    print(f"运行 WoSX (time_scheme={args.time_scheme}) ...")
    sim = Simulator(nx=nx, ny=ny, dt=dt, geometry=geo, time_scheme=args.time_scheme)
    sim.nmc = args.nmc
    for j in range(ny):
        for i in range(nx):
            sim.grid[0].set_vort(i, j, omega0[j, i])
            sim.grid[1].set_vort(i, j, omega0[j, i])

    t0 = time.perf_counter()
    num_steps = int(total_time / dt)
    for _ in range(num_steps):
        sim.step()
    wosx_time = time.perf_counter() - t0
    wosx_vort = sim.grid[sim.cur].vort.copy()
    wosx_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"  WoSX 耗时: {wosx_time:.2f}s")

    # ---------- Grid流函数法 ----------
    print("运行 Grid Stream Function...")
    grid_solver = GridStreamFunctionSolver(nx, ny, dt, dx, ox, oy, geo)
    grid_solver.set_omega(omega0)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        grid_solver.step()
    grid_time = time.perf_counter() - t0
    grid_vort = grid_solver.omega.copy()
    grid_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"  Grid 耗时: {grid_time:.2f}s")

    # ---------- 误差计算（域内） ----------
    def domain_mask():
        return np.array([[geo.inside_domain(Vec2(ox+(i+0.5)*dx, oy+(j+0.5)*dx))
                          for i in range(nx)] for j in range(ny)])

    mask = domain_mask()
    def domain_error(v1, v2):
        diff = (v1 - v2)[mask]
        ref = v2[mask]
        return float(np.linalg.norm(diff) / np.linalg.norm(ref))

    wosx_l2 = domain_error(wosx_vort, omega0)
    grid_l2 = domain_error(grid_vort, omega0)
    wosx_grid_diff = domain_error(wosx_vort, grid_vort)

    # ---------- 新增：环量与涡量变化评估 ----------
    Gamma0 = circulation(omega0, dx)
    L10 = l1_norm(omega0, dx)
    E0 = enstrophy(omega0, dx)
    Z0 = l2_norm(omega0, dx)

    Gamma_wosx = circulation(wosx_vort, dx)
    L1_wosx = l1_norm(wosx_vort, dx)
    E_wosx = enstrophy(wosx_vort, dx)
    Z_wosx = l2_norm(wosx_vort, dx)

    Gamma_grid = circulation(grid_vort, dx)
    L1_grid = l1_norm(grid_vort, dx)
    E_grid = enstrophy(grid_vort, dx)
    Z_grid = l2_norm(grid_vort, dx)

    def rel_change(new, old):
        return (new - old) / old * 100.0

    wosx_dGamma = rel_change(Gamma_wosx, Gamma0)
    grid_dGamma = rel_change(Gamma_grid, Gamma0)
    wosx_dL1 = rel_change(L1_wosx, L10)
    grid_dL1 = rel_change(L1_grid, L10)
    wosx_dE = rel_change(E_wosx, E0)
    grid_dE = rel_change(E_grid, E0)
    wosx_dZ = rel_change(Z_wosx, Z0)
    grid_dZ = rel_change(Z_grid, Z0)

    # ---------- 保存报告（含新指标） ----------
    report_path = os.path.join(output_dir, f"step2_wosx_vs_grid_report_{args.time_scheme}.txt")
    with open(report_path, "w") as f:
        f.write("Step2: WoSX vs Grid Stream Function 对比报告（含守恒量评估）\n")
        f.write(f"网格 {nx}x{ny}, dt={dt}, T={total_time}s, 障碍物半径={args.obstacle_radius}\n")
        f.write(f"时间积分方案: {args.time_scheme}\n\n")

        f.write("---- 基础误差 ----\n")
        f.write(f"WoSX  L2误差(相对初始): {wosx_l2:.6f}, 耗时: {wosx_time:.2f}s, 内存: {wosx_memory:.2f}MB\n")
        f.write(f"Grid  L2误差(相对初始): {grid_l2:.6f}, 耗时: {grid_time:.2f}s, 内存: {grid_memory:.2f}MB\n")
        f.write(f"WoSX与Grid差异: {wosx_grid_diff:.6f}\n\n")

        f.write("---- 守恒量评估 ----\n")
        f.write(f"初始总环量 Γ0 = {Gamma0:.6f}\n")
        f.write(f"初始 L1 范数 = {L10:.6f}\n")
        f.write(f"初始 Enstrophy = {E0:.6f}\n")
        f.write(f"初始 L2 范数 = {Z0:.6f}\n\n")

        f.write("WoSX:\n")
        f.write(f"  最终总环量 = {Gamma_wosx:.6f}，变化率 = {wosx_dGamma:+.2f}%\n")
        f.write(f"  最终 L1 范数 = {L1_wosx:.6f}，变化率 = {wosx_dL1:+.2f}%\n")
        f.write(f"  最终 Enstrophy = {E_wosx:.6f}，变化率 = {wosx_dE:+.2f}%\n")
        f.write(f"  最终 L2 范数 = {Z_wosx:.6f}，变化率 = {wosx_dZ:+.2f}%\n\n")

        f.write("Grid:\n")
        f.write(f"  最终总环量 = {Gamma_grid:.6f}，变化率 = {grid_dGamma:+.2f}%\n")
        f.write(f"  最终 L1 范数 = {L1_grid:.6f}，变化率 = {grid_dL1:+.2f}%\n")
        f.write(f"  最终 Enstrophy = {E_grid:.6f}，变化率 = {grid_dE:+.2f}%\n")
        f.write(f"  最终 L2 范数 = {Z_grid:.6f}，变化率 = {grid_dZ:+.2f}%\n\n")

        f.write("方法间守恒量差异 (WoSX - Grid):\n")
        f.write(f"  环量差异 = {Gamma_wosx - Gamma_grid:+.6f} (绝对), "
                f"{(Gamma_wosx - Gamma_grid)/Gamma0*100:+.2f}% (相对)\n")
        f.write(f"  L1范数差异 = {L1_wosx - L1_grid:+.6f}\n")
        f.write(f"  Enstrophy差异 = {E_wosx - E_grid:+.6f}\n")
        f.write(f"  L2范数差异 = {Z_wosx - Z_grid:+.6f}\n")

    print(f"\n报告已保存至 {report_path}")

    # ---------- 绘制对比图（附加守恒量柱状图） ----------
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 3, (1, 2))
    vmin, vmax = omega0.min(), omega0.max()
    img0 = ax1.imshow(omega0, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.set_title('Initial ω')
    fig.colorbar(img0, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(2, 3, 3)
    img1 = ax2.imshow(wosx_vort, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax2.set_title(f'WoSX ({args.time_scheme})')
    fig.colorbar(img1, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 3, 4)
    img2 = ax3.imshow(grid_vort, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax3.set_title('Grid')
    fig.colorbar(img2, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(2, 3, 5)
    diff_plot = np.abs(wosx_vort - grid_vort)
    img3 = ax4.imshow(diff_plot, origin='lower', cmap='hot')
    ax4.set_title('|WoSX - Grid|')
    fig.colorbar(img3, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(2, 3, 6)
    quantities = ['Γ', 'L1', 'Enstrophy', 'L2']
    wosx_changes = [wosx_dGamma, wosx_dL1, wosx_dE, wosx_dZ]
    grid_changes = [grid_dGamma, grid_dL1, grid_dE, grid_dZ]
    x = np.arange(len(quantities))
    width = 0.35
    bars1 = ax5.bar(x - width/2, wosx_changes, width, label='WoSX', color='skyblue')
    bars2 = ax5.bar(x + width/2, grid_changes, width, label='Grid', color='salmon')
    ax5.set_ylabel('Change (%)')
    ax5.set_title('Conserved quantity changes')
    ax5.set_xticks(x)
    ax5.set_xticklabels(quantities)
    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f"step2_wosx_vs_grid_{args.time_scheme}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"对比图已保存至 {fig_path}")

    # ---------- 终端摘要 ----------
    print("\n===== 守恒量摘要 =====")
    print(f"  环量变化: WoSX {wosx_dGamma:+.2f}%, Grid {grid_dGamma:+.2f}%")
    print(f"  Enstrophy变化: WoSX {wosx_dE:+.2f}%, Grid {grid_dE:+.2f}%")
    print(f"  L2范数变化: WoSX {wosx_dZ:+.2f}%, Grid {grid_dZ:+.2f}%")

if __name__ == '__main__':
    main()