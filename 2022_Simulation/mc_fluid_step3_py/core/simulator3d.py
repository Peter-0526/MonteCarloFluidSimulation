import math
import numpy as np
from numba import njit, prange
from .types3d import Vec3, RNG3D
from .grid3d import Grid3D, trilinear_interp_numba
from .biot_savart3d import BiotSavart3D


@njit(parallel=True)
def _numba_update_vorticity(nx, ny, nz, dx, ox, oy, oz,
                            dt, nu, nd, h, max_vel, max_vort,
                            vort_prev, vel_prev, vort_next,
                            periodic):
    """涡量更新：半拉格朗日平流 + 扩散 + 拉伸（支持周期边界）"""
    Lx = nx * dx
    Ly = ny * dx
    Lz = nz * dx

    for idx in prange(nx * ny * nz):
        k = idx // (nx * ny)
        j = (idx - k * nx * ny) // nx
        i = idx - k * nx * ny - j * nx

        cx = ox + (i + 0.5) * dx
        cy = oy + (j + 0.5) * dx
        cz = oz + (k + 0.5) * dx

        vx = vel_prev[k, j, i, 0]
        vy = vel_prev[k, j, i, 1]
        vz = vel_prev[k, j, i, 2]

        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed > max_vel:
            scale = max_vel / speed
            vx *= scale; vy *= scale; vz *= scale

        xb = cx - vx * dt
        yb = cy - vy * dt
        zb = cz - vz * dt

        # 周期环绕（替代 clamp）
        margin = dx
        if periodic:
            xb = ox + ((xb - ox) % Lx)
            yb = oy + ((yb - oy) % Ly)
            zb = oz + ((zb - oz) % Lz)
        else:
            margin = dx
            xb = min(max(xb, ox + margin), ox + nx*dx - margin)
            yb = min(max(yb, oy + margin), oy + ny*dx - margin)
            zb = min(max(zb, oz + margin), oz + nz*dx - margin)

        wx_avg, wy_avg, wz_avg = 0.0, 0.0, 0.0
        for _ in range(nd):
            xi_x = np.random.randn(); xi_y = np.random.randn(); xi_z = np.random.randn()
            scale_diff = math.sqrt(2.0 * nu * dt)
            xd = xb + scale_diff * xi_x
            yd = yb + scale_diff * xi_y
            zd = zb + scale_diff * xi_z
            if periodic:
                xd = ox + ((xd - ox) % Lx)
                yd = oy + ((yd - oy) % Ly)
                zd = oz + ((zd - oz) % Lz)
            else:
                xd = min(max(xd, ox + margin), ox + nx*dx - margin)
                yd = min(max(yd, oy + margin), oy + ny*dx - margin)
                zd = min(max(zd, oz + margin), oz + nz*dx - margin)

            w_prev = trilinear_interp_numba(vort_prev, xd, yd, zd,
                                             nx, ny, nz, dx, ox, oy, oz, periodic)

            norm_w = math.sqrt(w_prev[0]**2 + w_prev[1]**2 + w_prev[2]**2)
            if norm_w > 1e-12:
                dir_x = w_prev[0] / norm_w
                dir_y = w_prev[1] / norm_w
                dir_z = w_prev[2] / norm_w
                xp = xd + dir_x * h / 2; yp = yd + dir_y * h / 2; zp = zd + dir_z * h / 2
                xm = xd - dir_x * h / 2; ym = yd - dir_y * h / 2; zm = zd - dir_z * h / 2
                if periodic:
                    xp = ox + ((xp - ox) % Lx); yp = oy + ((yp - oy) % Ly); zp = oz + ((zp - oz) % Lz)
                    xm = ox + ((xm - ox) % Lx); ym = oy + ((ym - oy) % Ly); zm = oz + ((zm - oz) % Lz)
                else:
                    xp = min(max(xp, ox + margin), ox + nx*dx - margin)
                    yp = min(max(yp, oy + margin), oy + ny*dx - margin)
                    zp = min(max(zp, oz + margin), oz + nz*dx - margin)
                    xm = min(max(xm, ox + margin), ox + nx*dx - margin)
                    ym = min(max(ym, oy + margin), oy + ny*dx - margin)
                    zm = min(max(zm, oz + margin), oz + nz*dx - margin)

                vp = trilinear_interp_numba(vel_prev, xp, yp, zp,
                                             nx, ny, nz, dx, ox, oy, oz, periodic)
                vm = trilinear_interp_numba(vel_prev, xm, ym, zm,
                                             nx, ny, nz, dx, ox, oy, oz, periodic)
                diff_x = (vp[0] - vm[0]) * (norm_w / h)
                diff_y = (vp[1] - vm[1]) * (norm_w / h)
                diff_z = (vp[2] - vm[2]) * (norm_w / h)
            else:
                diff_x = diff_y = diff_z = 0.0

            stretch_mag = math.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            if stretch_mag > 10.0:
                s = 10.0 / stretch_mag
                diff_x *= s; diff_y *= s; diff_z *= s

            wx_avg += w_prev[0] + dt * diff_x
            wy_avg += w_prev[1] + dt * diff_y
            wz_avg += w_prev[2] + dt * diff_z

        wx_avg /= nd; wy_avg /= nd; wz_avg /= nd

        w_mag = math.sqrt(wx_avg**2 + wy_avg**2 + wz_avg**2)
        if w_mag > max_vort:
            s = max_vort / w_mag
            wx_avg *= s; wy_avg *= s; wz_avg *= s

        vort_next[k, j, i, 0] = wx_avg
        vort_next[k, j, i, 1] = wy_avg
        vort_next[k, j, i, 2] = wz_avg


class Simulator3D:
    def __init__(self, nx, ny, nz, dt, nu=0.0, L=1.0,
                 use_control_variate=False, initial_nmc=None,
                 use_periodic=False):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dt = dt
        self.nu = nu
        self.L = L
        self.use_control_variate = use_control_variate
        self.use_periodic = use_periodic

        self.dx = (2.0 * L) / max(nx, ny, nz)
        self.ox = self.oy = self.oz = -L
        self.grid = [Grid3D(nx, ny, nz, self.dx, self.ox, self.oy, self.oz,
                            periodic=use_periodic) for _ in range(2)]
        self.cur = 0

        self.nmc = 16
        self.nd = 4
        self.h = self.dx
        self.rng = RNG3D()

        # 控制变量缓存
        self.prev_vort = None
        self.prev_vel = None
        self.tmp_grid = Grid3D(nx, ny, nz, self.dx, self.ox, self.oy, self.oz,
                               periodic=use_periodic)

        # 第一步高采样数：默认 4 倍正常 nmc，可通过 initial_nmc 显式设置
        self.initial_nmc = initial_nmc if initial_nmc is not None else self.nmc * 4

        self._init_vorticity()

    def _init_vorticity(self):
        # 默认生成两个高斯涡，但在 TG 示例中会被覆盖。
        sigma = 0.3
        centers = [Vec3(-0.4, 0.0, 0.0), Vec3(0.4, 0.0, 0.0)]
        strength = 0.8
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    p = self.grid[0].grid_pos(i, j, k)
                    w = Vec3()
                    for c in centers:
                        r = p - c
                        dist = r.norm()
                        envelope = math.exp(-dist * dist / (2 * sigma * sigma))
                        perp = Vec3(0.0, -r.z, r.y)
                        if perp.norm() > 1e-8:
                            perp = perp * (1.0 / perp.norm())
                        w = w + perp * (strength * envelope)
                    self.grid[0].set_vort(i, j, k, w)
                    self.grid[1].set_vort(i, j, k, w)

    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1 - self.cur]
        nx, ny, nz = self.nx, self.ny, self.nz

        # ---------- 收集所有网格点坐标 ----------
        xs = np.zeros((nx * ny * nz, 3))
        idx = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    p = prev.grid_pos(i, j, k)
                    xs[idx, 0] = p.x; xs[idx, 1] = p.y; xs[idx, 2] = p.z
                    idx += 1

        # ---------- 速度估计 ----------
        if self.use_control_variate and self.prev_vort is not None:
            # 控制变量模式：估计涡量增量的速度，并加上前一步速度
            delta_vort = prev.vort - self.prev_vort
            self.tmp_grid.vort = delta_vort
            delta_v = BiotSavart3D.estimate_velocity_batch(
                self.tmp_grid, xs, self.nmc, use_importance=True,
                periodic=self.use_periodic,
                Lx=nx*self.dx, Ly=ny*self.dx, Lz=nz*self.dx)
            base_vel = self.prev_vel if self.prev_vel is not None else prev.vel
            velocities = delta_v + base_vel.reshape(-1, 3)
        else:
            # 第一步（或未启用控制变量）：全速度估计
            if self.use_control_variate and self.prev_vort is None:
                nmc_eff = self.initial_nmc
            else:
                nmc_eff = self.nmc
            velocities = BiotSavart3D.estimate_velocity_batch(
                prev, xs, nmc_eff, use_importance=True,
                periodic=self.use_periodic,
                Lx=nx*self.dx, Ly=ny*self.dx, Lz=nz*self.dx)

        # 速度限幅
        speeds = np.linalg.norm(velocities, axis=1)
        max_v = 2.0
        mask = speeds > max_v
        if mask.any():
            velocities[mask] *= (max_v / speeds[mask])[:, None]

        # 写回速度场
        prev.vel = velocities.reshape((nz, ny, nx, 3))

        # 更新控制变量缓存
        if self.use_control_variate:
            self.prev_vel = prev.vel.copy()
            self.prev_vort = prev.vort.copy()

        # ---------- 涡量更新 ----------
        _numba_update_vorticity(
            nx, ny, nz, self.dx, self.ox, self.oy, self.oz,
            self.dt, self.nu, self.nd, self.h, 2.0, 5.0,
            prev.vort, prev.vel, nxt.vort,
            self.use_periodic)

        self.cur = 1 - self.cur

    @property
    def vorticity_slice(self):
        k0 = self.nz // 2
        return np.sqrt(self.grid[self.cur].vort[k0, :, :, 0] ** 2 +
                       self.grid[self.cur].vort[k0, :, :, 1] ** 2 +
                       self.grid[self.cur].vort[k0, :, :, 2] ** 2)