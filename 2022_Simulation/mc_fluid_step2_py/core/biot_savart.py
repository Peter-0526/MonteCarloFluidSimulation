# core/biot_savart.py
import numpy as np
from numba import njit, prange
from .types import Vec2


# ---------------- Numba 核函数（nopython 模式） ----------------

@njit(cache=True)
def _kernel(x, y, sx, sy):
    """2D Biot–Savart 核: k = (r.y, -r.x) / (2*pi*r^2), r = sample - x
    返回 (kx, ky)，全部用 float64 标量，避免 Vec2 对象开销。"""
    rx = sx - x
    ry = sy - y
    r2 = rx * rx + ry * ry + 1e-12
    kx = ry / (2.0 * np.pi * r2)
    ky = -rx / (2.0 * np.pi * r2)
    return kx, ky


@njit(cache=True)
def _get_vort(vort, nx, ny, dx, ox, oy, px, py):
    """与 Grid.get_vort 等价的双线性插值（直接操作 ndarray）。"""
    fx = (px - ox) / dx - 0.5
    fy = (py - oy) / dx - 0.5
    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0
    if i0 < 0:
        i0 = 0
    elif i0 > nx - 2:
        i0 = nx - 2
    if j0 < 0:
        j0 = 0
    elif j0 > ny - 2:
        j0 = ny - 2
    i1, j1 = i0 + 1, j0 + 1
    v00 = vort[j0, i0]
    v10 = vort[j0, i1]
    v01 = vort[j1, i0]
    v11 = vort[j1, i1]
    return ((1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) +
            ty * ((1.0 - tx) * v01 + tx * v11))


@njit(cache=True)
def _estimate_velocity(vort, nx, ny, dx, ox, oy, x, y, nsamples, area):
    """单点 Monte-Carlo 速度估计（串行版，保留给兼容）。"""
    xmax = ox + nx * dx
    ymax = oy + ny * dx
    pdf = 1.0 / area
    sum_vx = 0.0
    sum_vy = 0.0
    for _ in range(nsamples):
        sx = np.random.uniform(ox, xmax)
        sy = np.random.uniform(oy, ymax)
        w = _get_vort(vort, nx, ny, dx, ox, oy, sx, sy)
        kx, ky = _kernel(x, y, sx, sy)
        sum_vx += kx * (w / pdf)
        sum_vy += ky * (w / pdf)
    return sum_vx / nsamples, sum_vy / nsamples


@njit(cache=True, parallel=True)
def _estimate_velocity_batch(vort, nx, ny, dx, ox, oy,
                             xs, ys, sx_all, sy_all, area,
                             out_vx, out_vy):
    """批量版：所有网格点共享同一组采样点，prange 并行。
    样本在调用方预先用 numpy 生成，避开 prange 内 RNG 竞态。"""
    n = xs.size
    nsamples = sx_all.size
    pdf = 1.0 / area
    for idx in prange(n):
        x = xs[idx]
        y = ys[idx]
        svx = 0.0
        svy = 0.0
        for k in range(nsamples):
            w = _get_vort(vort, nx, ny, dx, ox, oy, sx_all[k], sy_all[k])
            kx, ky = _kernel(x, y, sx_all[k], sy_all[k])
            svx += kx * (w / pdf)
            svy += ky * (w / pdf)
        out_vx[idx] = svx / nsamples
        out_vy[idx] = svy / nsamples


def estimate_velocity_batch(prev_vort, xs, ys, nsamples, seed=None):
    """
    批量估计所有点的速度（使用公共随机样本并并行计算）。

    参数
    -----
    prev_vort : Grid
        涡量网格对象。
    xs, ys : array_like, float64
        所有目标点的 x 和 y 坐标（一维数组，长度相同）。
    nsamples : int
        每个点的蒙特卡洛采样数。
    seed : int, optional
        随机种子，用于可复现。

    返回
    ------
    vx, vy : ndarray, shape (len(xs),)
        速度分量。
    """
    xs = np.ascontiguousarray(xs, dtype=np.float64)
    ys = np.ascontiguousarray(ys, dtype=np.float64)
    if seed is not None:
        np.random.seed(seed)

    # 生成公共采样点（Numba 内部 RNG 无法在并行中安全使用，故在外部生成）
    sx_all = np.random.uniform(prev_vort.ox,
                               prev_vort.ox + prev_vort.nx * prev_vort.dx,
                               nsamples)
    sy_all = np.random.uniform(prev_vort.oy,
                               prev_vort.oy + prev_vort.ny * prev_vort.dx,
                               nsamples)

    out_vx = np.empty(xs.size)
    out_vy = np.empty(ys.size)
    _estimate_velocity_batch(prev_vort.vort, prev_vort.nx, prev_vort.ny,
                             prev_vort.dx, prev_vort.ox, prev_vort.oy,
                             xs, ys, sx_all, sy_all, prev_vort.area,
                             out_vx, out_vy)
    return out_vx, out_vy


# ---------------- 对外接口（保持原签名，simulator.py 无需改动） ----------------

class BiotSavart:
    @staticmethod
    def kernel(x: Vec2, y: Vec2) -> Vec2:
        kx, ky = _kernel(x.x, x.y, y.x, y.y)
        return Vec2(kx, ky)

    @staticmethod
    def estimate_velocity(prev_vort, x: Vec2, nsamples: int, rng=None) -> Vec2:
        # rng 参数保留仅为兼容旧调用；真正的随机源是 Numba 的 np.random
        vx, vy = _estimate_velocity(
            prev_vort.vort, prev_vort.nx, prev_vort.ny, prev_vort.dx,
            prev_vort.ox, prev_vort.oy,
            x.x, x.y, nsamples, prev_vort.area)
        return Vec2(vx, vy)