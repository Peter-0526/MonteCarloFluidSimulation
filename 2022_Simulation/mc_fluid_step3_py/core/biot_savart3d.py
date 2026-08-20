import math
import numpy as np
from numba import njit, prange
from typing import Optional
from .types3d import Vec3, RNG3D
from .grid3d import Grid3D


@njit(parallel=True)
def _numba_biot_savart_batch(vort_samples, sample_points, pdfs, query_points,
                             periodic, Lx, Ly, Lz):
    M = query_points.shape[0]
    ns = sample_points.shape[0]
    v = np.zeros((M, 3), dtype=np.float64)

    for i in prange(M):
        x = query_points[i, 0]
        y = query_points[i, 1]
        z = query_points[i, 2]

        total_x = 0.0
        total_y = 0.0
        total_z = 0.0

        for j in range(ns):
            rx0 = sample_points[j, 0] - x
            ry0 = sample_points[j, 1] - y
            rz0 = sample_points[j, 2] - z

            if periodic:
                # 周期镜像（3×3×3）
                for ix in (-1, 0, 1):
                    rx = rx0 + ix * Lx
                    for iy in (-1, 0, 1):
                        ry = ry0 + iy * Ly
                        for iz in (-1, 0, 1):
                            rz = rz0 + iz * Lz
                            r2 = rx*rx + ry*ry + rz*rz + 1e-12
                            r = np.sqrt(r2)
                            factor = 1.0 / (4.0 * np.pi * r2 * r)

                            gx = rx * factor
                            gy = ry * factor
                            gz = rz * factor

                            wx = vort_samples[j, 0]
                            wy = vort_samples[j, 1]
                            wz = vort_samples[j, 2]

                            # 修正符号：u = (1/(4π)) ∫ ω × (x-y)/r^3
                            # 但这里 g = (y-x)/r^3，所以取负叉积
                            cross_x = wz * gy - wy * gz
                            cross_y = wx * gz - wz * gx
                            cross_z = wy * gx - wx * gy

                            w = 1.0 / pdfs[j]
                            total_x += cross_x * w
                            total_y += cross_y * w
                            total_z += cross_z * w
            else:
                r2 = rx0*rx0 + ry0*ry0 + rz0*rz0 + 1e-12
                r = np.sqrt(r2)
                factor = 1.0 / (4.0 * np.pi * r2 * r)

                gx = rx0 * factor
                gy = ry0 * factor
                gz = rz0 * factor

                wx = vort_samples[j, 0]
                wy = vort_samples[j, 1]
                wz = vort_samples[j, 2]

                cross_x = wz * gy - wy * gz
                cross_y = wx * gz - wz * gx
                cross_z = wy * gx - wx * gy

                w = 1.0 / pdfs[j]
                total_x += cross_x * w
                total_y += cross_y * w
                total_z += cross_z * w

        v[i, 0] = total_x / ns
        v[i, 1] = total_y / ns
        v[i, 2] = total_z / ns

    return v


class BiotSavart3D:
    @staticmethod
    def kernel(x: Vec3, y: Vec3) -> Vec3:
        # 注意：此处返回 y - x，配合外部叉积需要负号
        r = y - x
        r2 = r.norm2() + 1e-12
        factor = 1.0 / (4.0 * math.pi * r2 * math.sqrt(r2))
        return Vec3(r.x * factor, r.y * factor, r.z * factor)

    # ------------------------------------------------------------------
    # 构建涡量幅值的 CDF（用于重要性采样）
    # ------------------------------------------------------------------
    @staticmethod
    def build_vorticity_cdf(vort_grid: Grid3D):
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        mag = np.sqrt(np.sum(vort_grid.vort**2, axis=-1))
        weights = mag.flatten()
        total = weights.sum()
        if total < 1e-12:
            weights = np.ones_like(weights)
            weights /= weights.sum()
        else:
            weights /= total
        cdf = np.cumsum(weights)
        indices = np.arange(nx * ny * nz)
        return weights, cdf, indices

    @staticmethod
    def importance_sample(vort_grid: Grid3D, nsamples: int):
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz

        weights, cdf, indices = BiotSavart3D.build_vorticity_cdf(vort_grid)

        r = np.random.uniform(0.0, 1.0, size=nsamples)
        idx = np.searchsorted(cdf, r)
        idx = np.clip(idx, 0, len(indices) - 1)

        k = idx // (nx * ny)
        j = (idx - k * nx * ny) // nx
        i = idx - k * nx * ny - j * nx

        xi = np.random.uniform(-0.5, 0.5, size=(nsamples, 3))
        points = np.zeros((nsamples, 3), dtype=np.float64)
        points[:, 0] = ox + (i + 0.5 + xi[:, 0]) * dx
        points[:, 1] = oy + (j + 0.5 + xi[:, 1]) * dx
        points[:, 2] = oz + (k + 0.5 + xi[:, 2]) * dx

        cell_vol = dx**3
        pdfs = weights[idx] / cell_vol
        pdfs = np.maximum(pdfs, 1e-20)

        return points, pdfs

    @staticmethod
    def uniform_sample(vort_grid: Grid3D, nsamples: int):
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz
        points = np.random.uniform(
            [ox, oy, oz],
            [ox + nx*dx, oy + ny*dx, oz + nz*dx],
            size=(nsamples, 3)
        )
        volume = nx * ny * nz * dx**3
        pdfs = np.full(nsamples, 1.0 / volume)
        return points, pdfs

    # ------------------------------------------------------------------
    # 批量估计速度（支持周期性）
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_velocity_batch(vort_grid: Grid3D, points: np.ndarray,
                                nsamples: int, use_importance: bool = True,
                                periodic: bool = False, Lx: Optional[float] = None,
                                Ly: Optional[float] = None, Lz: Optional[float] = None):
        """
        批量估计速度（向量化 + 重要性采样 + Numba 并行）
        points: (M,3) 查询点坐标
        nsamples: 样本数（全局共享）
        use_importance: 是否按涡量幅值采样
        periodic: 是否使用周期镜像
        Lx, Ly, Lz: 周期域尺寸（需在 periodic=True 时提供）
        返回: (M,3) 速度数组
        """
        points = np.ascontiguousarray(points, dtype=np.float64)
        M = points.shape[0]

        if periodic and (Lx is None or Ly is None or Lz is None):
            raise ValueError("Periodic mode requires Lx, Ly, Lz")

        # 生成全局共享样本点
        if use_importance:
            y_points, pdfs = BiotSavart3D.importance_sample(vort_grid, nsamples)
        else:
            y_points, pdfs = BiotSavart3D.uniform_sample(vort_grid, nsamples)

        # 批量插值样本点上的涡量
        omega = vort_grid.get_vort_interp_batch(y_points[:, 0],
                                                y_points[:, 1],
                                                y_points[:, 2])

        # 转为连续 numpy 数组
        y_points = np.ascontiguousarray(y_points, dtype=np.float64)
        pdfs = np.ascontiguousarray(pdfs, dtype=np.float64)
        omega = np.ascontiguousarray(omega, dtype=np.float64)

        # 调用 Numba 并行函数
        velocities = _numba_biot_savart_batch(
            omega, y_points, pdfs, points,
            periodic,
            Lx if periodic else 0.0,
            Ly if periodic else 0.0,
            Lz if periodic else 0.0
        )

        return velocities

    # ------------------------------------------------------------------
    # 原逐点估计（保留，供自定义使用）
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_velocity(vort_grid: Grid3D, x: Vec3, nsamples: int,
                          rng: RNG3D) -> Vec3:
        nx, ny, nz = vort_grid.nx, vort_grid.ny, vort_grid.nz
        dx = vort_grid.dx
        ox, oy, oz = vort_grid.ox, vort_grid.oy, vort_grid.oz
        total = Vec3()
        pdf = 1.0 / (nx * ny * nz * dx**3)
        for _ in range(nsamples):
            y = Vec3(rng.uniform(ox, ox + nx * dx),
                     rng.uniform(oy, oy + ny * dx),
                     rng.uniform(oz, oz + nz * dx))
            w = vort_grid.get_vort_interp(y)
            k = BiotSavart3D.kernel(x, y)
            cross = w.cross(k)
            total = total + cross * (-1.0 / pdf)   # 修正符号
        return total * (1.0 / nsamples)