import sys, os, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.simulator3d import Simulator3D


def tg_initial_vorticity(x, y, z, k):
    wx = -k * np.cos(k * x) * np.sin(k * y) * np.sin(k * z)
    wy = -k * np.sin(k * x) * np.cos(k * y) * np.sin(k * z)
    wz =  2.0 * k * np.sin(k * x) * np.sin(k * y) * np.cos(k * z)
    return wx, wy, wz


def main():
    parser = argparse.ArgumentParser(description='3D Taylor-Green benchmark (Re=1600, MC)')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--total_time', type=float, default=20.0)
    parser.add_argument('--nmc', type=int, default=1024)
    parser.add_argument('--nu', type=float, default=1.0 / 1600.0)
    parser.add_argument('--L', type=float, default=np.pi)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--ref_file', type=str, default='ref.txt')
    parser.add_argument('--control_variate', action='store_true')
    parser.add_argument('--initial_nmc', type=int, default=None)
    args = parser.parse_args()

    nx = ny = nz = args.nx
    dt, total_time = args.dt, args.total_time
    L = args.L
    k = np.pi / L
    dx = (2.0 * L) / nx
    ox = oy = oz = -L
    V = (2.0 * L) ** 3

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sim = Simulator3D(
        nx=nx, ny=ny, nz=nz, dt=dt, nu=args.nu, L=L,
        use_control_variate=args.control_variate,
        initial_nmc=args.initial_nmc,
        use_periodic=True
    )
    sim.nmc = args.nmc

    # 仅初始化涡量场（t=0 解析值）
    from core.types3d import Vec3
    for kk in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = ox + (i + 0.5) * dx
                y = oy + (j + 0.5) * dx
                z = oz + (kk + 0.5) * dx
                wx, wy, wz = tg_initial_vorticity(x, y, z, k)
                sim.grid[0].set_vort(i, j, kk, Vec3(wx, wy, wz))
                sim.grid[1].set_vort(i, j, kk, Vec3(wx, wy, wz))

    # 不计算初始速度/动能，时间序列从第一步开始
    num_steps = int(total_time / dt)
    times = []
    kinetic_energy = []
    enstrophy = []

    k0 = nz // 2
    frame_times = []
    sim_frames = []

    record_interval = max(1, num_steps // 30)

    print(f"开始模拟: {nx}³, dt={dt}, total_time={total_time}, ν={args.nu:.6f}, nmc={args.nmc}")
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        sim.step()
        elapsed = time.perf_counter() - t0
        t = step * dt

        # 读取当前时刻的速度和涡量
        vort = sim.grid[sim.cur].vort       # (nz, ny, nx, 3)
        vel  = sim.grid[1 - sim.cur].vel    # 速度存在旧网格

        Ek = 0.5 * np.sum(vel**2) * (dx**3) / V
        Z  = 0.5 * np.sum(vort**2) * (dx**3) / V

        times.append(t)
        kinetic_energy.append(Ek)
        enstrophy.append(Z)

        if step % record_interval == 0 or step == num_steps:
            sim_frames.append(np.sqrt(np.sum(vort[k0]**2, axis=-1)))
            frame_times.append(t)

        if step % max(1, num_steps // 10) == 0:
            print(f"  Step {step}/{num_steps} (t={t:.2f}, 耗时 {elapsed:.2f}s): "
                  f"Ek={Ek:.6f}, Z={Z:.6f}")

    # 保存报告（不含 t=0）
    report_path = os.path.join(output_dir, "tg3d_report.txt")
    with open(report_path, "w") as f:
        f.write("3D Taylor-Green standard benchmark (Re=1600, MC velocity)\n")
        f.write(f"网格 {nx}³, L={L:.6f}, k={k:.6f}, dt={dt}, "
                f"total_time={total_time}, ν={args.nu:.8f}, nmc={args.nmc}\n\n")
        f.write("时间(t)\t平均动能Ek\t拟涡能Z\n")
        for i, t in enumerate(times):
            f.write(f"{t:.6f}\t{kinetic_energy[i]:.8f}\t{enstrophy[i]:.8f}\n")
    print(f"\n统计报告已保存至 {report_path}")

    # 绘制 E(t) 和 Z(t)（只显示 t>=dt 的数据）
    times = np.array(times)
    kinetic_energy = np.array(kinetic_energy)
    enstrophy = np.array(enstrophy)

    plt.figure(figsize=(8, 5))
    plt.plot(times, kinetic_energy, 'o-', markersize=3)
    plt.xlabel('Time t'); plt.ylabel(r'Kinetic energy $E_k$')
    plt.title(f'3D TG Kinetic Energy (Re=1600, {nx}³, MC)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "tg3d_kinetic_energy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(times, enstrophy, 's-', markersize=3)
    plt.xlabel('Time t'); plt.ylabel(r'Enstrophy $Z$')
    plt.title(f'3D TG Enstrophy (Re=1600, {nx}³, MC)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "tg3d_enstrophy.png"), dpi=150)
    plt.close()

    # 与参考数据对比（如果提供）
    if args.ref_file and os.path.exists(args.ref_file):
        data = np.loadtxt(args.ref_file, comments='#')
        t_ref, E_ref, Z_ref = data[:, 0], data[:, 1], data[:, 2]

        plt.figure(figsize=(8, 5))
        plt.plot(times, kinetic_energy, 'o-', label='MC simulation', markersize=3)
        plt.plot(t_ref, E_ref, 's--', label='OpenSBLI reference (64³)', markersize=3)
        plt.xlabel('Time t'); plt.ylabel(r'Kinetic energy $E_k$')
        plt.title(f'3D TG Kinetic Energy (Re=1600, {nx}³)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, "tg3d_Ek_comparison.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(times, enstrophy, 'o-', label='MC simulation', markersize=3)
        plt.plot(t_ref, Z_ref, 's--', label='OpenSBLI reference (64³)', markersize=3)
        plt.xlabel('Time t'); plt.ylabel(r'Enstrophy $Z$')
        plt.title(f'3D TG Enstrophy (Re=1600, {nx}³)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, "tg3d_Z_comparison.png"), dpi=150)
        plt.close()

        # 峰值误差统计
        idx_ref = np.argmax(Z_ref)
        Z_ref_peak = Z_ref[idx_ref]
        t_ref_peak = t_ref[idx_ref]

        # 若模拟时间未覆盖参考峰值，只比较已有区间
        if len(enstrophy) > 0:
            idx_sim = np.argmax(enstrophy)
            Z_sim_peak = enstrophy[idx_sim]
            t_sim_peak = times[idx_sim]

            err_peak = abs(Z_sim_peak - Z_ref_peak) / Z_ref_peak * 100.0
            err_tpeak = abs(t_sim_peak - t_ref_peak) / t_ref_peak * 100.0

            print(f"\n参考峰值: Z={Z_ref_peak:.4f} @ t={t_ref_peak:.2f}")
            print(f"模拟峰值: Z={Z_sim_peak:.4f} @ t={t_sim_peak:.2f}")
            print(f"峰值拟涡能相对误差 = {err_peak:.2f}%")
            print(f"峰值时刻相对误差   = {err_tpeak:.2f}%")

            with open(report_path, "a") as f:
                f.write(f"\n参考峰值: Z={Z_ref_peak:.6f} @ t={t_ref_peak:.4f}\n")
                f.write(f"模拟峰值: Z={Z_sim_peak:.6f} @ t={t_sim_peak:.4f}\n")
                f.write(f"峰值拟涡能相对误差 = {err_peak:.4f}%\n")
                f.write(f"峰值时刻相对误差   = {err_tpeak:.4f}%\n")
        else:
            print("无模拟数据，无法计算峰值误差")
    else:
        print(f"警告: 参考文件 {args.ref_file} 不存在，跳过对比绘图")

    print("完成，输出已保存至", output_dir)


if __name__ == '__main__':
    main()