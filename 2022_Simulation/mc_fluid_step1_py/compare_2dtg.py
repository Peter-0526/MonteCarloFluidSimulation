import argparse, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.types import Vec2, RNG
from core.grid import Grid
from core.biot_savart import biot_savart_velocity_numba
from core.simulator import Simulator


def tg_vorticity_0(x, y, k):
    return 2.0 * k * np.sin(k * x) * np.sin(k * y)

def tg_velocity(x, y, k):
    u = np.sin(k * x) * np.cos(k * y)
    v = -np.cos(k * x) * np.sin(k * y)
    return u, v


def main():
    parser = argparse.ArgumentParser(description='2D Taylor-Green 严格验证（Numba 加速）')
    parser.add_argument('--nx', type=int, default=32)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--total_time', type=float, default=0.5)
    parser.add_argument('--nu', type=float, default=0.05)
    parser.add_argument('--L', type=float, default=np.pi)
    parser.add_argument('--nmc', type=int, default=2000)
    parser.add_argument('--nd', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    L = args.L
    k = np.pi / L
    dx = 2 * L / args.nx
    ox = oy = -L
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1. 静态验证 ==========
    print("=== 静态验证：Biot-Savart 速度重建 ===")
    grid = Grid(args.nx, args.nx, dx, ox, oy)
    for j in range(args.nx):
        for i in range(args.nx):
            p = grid.grid_pos(i, j)
            grid.set_vort(i, j, tg_vorticity_0(p.x, p.y, k))

    xs = np.array([grid.grid_pos(i, j).x for j in range(args.nx) for i in range(args.nx)])
    ys = np.array([grid.grid_pos(i, j).y for j in range(args.nx) for i in range(args.nx)])
    vel = biot_savart_velocity_numba(
        grid.vort, args.nx, args.nx, dx, ox, oy,
        xs, ys, args.nmc, True, args.nx * dx, args.nx * dx, seed=42)

    u_ref = np.array([tg_velocity(x, y, k)[0] for x, y in zip(xs, ys)])
    v_ref = np.array([tg_velocity(x, y, k)[1] for x, y in zip(xs, ys)])
    u_err2 = np.sum((vel[:, 0] - u_ref) ** 2)
    v_err2 = np.sum((vel[:, 1] - v_ref) ** 2)
    ref2 = np.sum(u_ref ** 2 + v_ref ** 2)
    vel_l2 = np.sqrt((u_err2 + v_err2) / ref2)
    print(f"  速度 L2 相对误差 = {vel_l2:.6f}")

    # ============================================================
    # 速度场可视化（新增）
    # ============================================================
    # 将速度 reshape 成二维网格
    vel_u_2d = vel[:, 0].reshape((args.nx, args.nx))
    vel_v_2d = vel[:, 1].reshape((args.nx, args.nx))
    u_ref_2d = u_ref.reshape((args.nx, args.nx))
    v_ref_2d = v_ref.reshape((args.nx, args.nx))

    # 网格点坐标（用于 quiver）
    i_indices = np.arange(args.nx)
    j_indices = np.arange(args.nx)
    X, Y = np.meshgrid(ox + (i_indices + 0.5) * dx,
                    oy + (j_indices + 0.5) * dx, indexing='ij')

    # 1. 速度幅值对比云图
    mag_mc  = np.sqrt(vel_u_2d**2 + vel_v_2d**2)
    mag_ref = np.sqrt(u_ref_2d**2 + v_ref_2d**2)
    vmin = min(mag_mc.min(), mag_ref.min())
    vmax = max(mag_mc.max(), mag_ref.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(mag_mc.T, origin='lower', cmap='YlGnBu',
                        vmin=vmin, vmax=vmax,
                        extent=[ox, ox + args.nx*dx, oy, oy + args.nx*dx])
    axes[0].set_title('MC Velocity Magnitude')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(mag_ref.T, origin='lower', cmap='YlGnBu',
                        vmin=vmin, vmax=vmax,
                        extent=[ox, ox + args.nx*dx, oy, oy + args.nx*dx])
    axes[1].set_title('Exact Velocity Magnitude')
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'tg2d_velocity_magnitude.png'), dpi=150)
    plt.close()
    print("速度幅值对比图已保存：tg2d_velocity_magnitude.png")

    # 2. 速度矢量场对比（quiver）
    step = max(1, args.nx // 16)  # 下采样步长

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].quiver(X[::step, ::step], Y[::step, ::step],
                vel_u_2d[::step, ::step].T, vel_v_2d[::step, ::step].T)
    axes[0].set_title('MC Velocity Field')
    axes[0].set_aspect('equal')
    axes[1].quiver(X[::step, ::step], Y[::step, ::step],
                u_ref_2d[::step, ::step].T, v_ref_2d[::step, ::step].T)
    axes[1].set_title('Exact Velocity Field')
    axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'tg2d_velocity_quiver.png'), dpi=150)
    plt.close()
    print("速度矢量对比图已保存：tg2d_velocity_quiver.png")

    # ========== 2. 动态验证 ==========
    print("=== 动态验证：涡量演化 ===")
    sim = Simulator(args.nx, args.nx, args.dt, nu=args.nu, L=L,
                    periodic=True, nmc=args.nmc, nd=args.nd)
    sim.set_vorticity_field(lambda p: tg_vorticity_0(p.x, p.y, k))

    num_steps = int(args.total_time / args.dt)
    times = [0.0]; l2_errors = [0.0]
    frame_times = []; sim_frames = []; ref_frames = []
    record_interval = max(1, num_steps // 20)

    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        sim.step()
        t = step * args.dt

        vort = sim.vorticity
        diff2 = 0.0; ref2 = 0.0
        for j in range(args.nx):
            for i in range(args.nx):
                p = sim.grid[sim.cur].grid_pos(i, j)
                w_ref = tg_vorticity_0(p.x, p.y, k) * np.exp(-2 * args.nu * k * k * t)
                w_sim = vort[j, i]
                diff2 += (w_sim - w_ref) ** 2
                ref2 += w_ref ** 2
        l2 = np.sqrt(diff2 / ref2) if ref2 > 0 else 0.0
        times.append(t); l2_errors.append(l2)

        if step % record_interval == 0 or step == num_steps:
            # 保存整个二维场（修复 imshow 错误）
            sim_slice = sim.vorticity.copy()
            ref_slice = np.zeros((args.nx, args.nx))
            for j in range(args.nx):
                for i in range(args.nx):
                    p = sim.grid[sim.cur].grid_pos(i, j)
                    ref_slice[j, i] = (tg_vorticity_0(p.x, p.y, k)
                                       * np.exp(-2 * args.nu * k * k * t))
            sim_frames.append(sim_slice)
            ref_frames.append(ref_slice)
            frame_times.append(t)

        if step % max(1, num_steps // 10) == 0:
            print(f"  Step {step}/{num_steps} (t={t:.3f}, "
                  f"耗时 {time.perf_counter() - t0:.2f}s): L2={l2:.6f}")

    # ---------- 误差曲线 ----------
    plt.figure(figsize=(8, 5))
    plt.semilogy(times, l2_errors, 'o-')
    plt.xlabel('Time (s)'); plt.ylabel('L2 relative error')
    plt.title(f'2D TG (ν={args.nu}, k={k:.3f}): MC vs Exact')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'tg2d_error.png'), dpi=150)
    plt.close()

    # ---------- 最终全场对比（2D） ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax_abs = max(np.max(np.abs(sim_frames[-1])), np.max(np.abs(ref_frames[-1])))
    vmin, vmax = -vmax_abs, vmax_abs

    im0 = axes[0].imshow(sim_frames[-1], origin='lower', cmap='RdBu',
                        vmin=vmin, vmax=vmax,
                        extent=[ox, ox + args.nx * dx, oy, oy + args.nx * dx])
    axes[0].set_title(f'MC Vorticity (t={args.total_time:.2f})')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ref_frames[-1], origin='lower', cmap='RdBu',
                        vmin=vmin, vmax=vmax,
                        extent=[ox, ox + args.nx * dx, oy, oy + args.nx * dx])
    axes[1].set_title('Exact Vorticity')
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'tg2d_slice.png'), dpi=150)
    plt.close()

    # ---------- 报告 ----------
    with open(os.path.join(args.output_dir, 'tg2d_report.txt'), 'w') as f:
        f.write("2D Taylor-Green verification (Numba, fixed RNG)\n")
        f.write(f"nx={args.nx}, dt={args.dt}, nu={args.nu}, "
                f"nmc={args.nmc}, nd={args.nd}\n")
        f.write(f"Velocity L2 error (Biot-Savart) = {vel_l2:.6f}\n")
        f.write("Time(s)\tL2_error\n")
        for i, t in enumerate(times):
            f.write(f"{t:.4f}\t{l2_errors[i]:.8f}\n")

    print("完成，输出已保存至", args.output_dir)


if __name__ == '__main__':
    main()