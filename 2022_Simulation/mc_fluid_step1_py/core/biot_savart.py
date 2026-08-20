import numpy as np
from numba import njit, prange


@njit(cache=True)
def interp_vort_numba(vort, nx, ny, dx, ox, oy, x, y, periodic):
    fx = (x - ox) / dx - 0.5
    fy = (y - oy) / dx - 0.5
    if periodic:
        fx = fx % nx
        fy = fy % ny
        i0 = int(fx); j0 = int(fy)
        i1 = (i0 + 1) % nx
        j1 = (j0 + 1) % ny
    else:
        i0 = int(fx); j0 = int(fy)
        i0 = max(0, min(i0, nx - 2))
        j0 = max(0, min(j0, ny - 2))
        i1 = i0 + 1; j1 = j0 + 1
    tx = fx - i0; ty = fy - j0
    v00 = vort[j0, i0]; v10 = vort[j0, i1]
    v01 = vort[j1, i0]; v11 = vort[j1, i1]
    return ((1 - ty) * ((1 - tx) * v00 + tx * v10) +
            ty * ((1 - tx) * v01 + tx * v11))


@njit(parallel=True, cache=True)
def biot_savart_velocity_numba(vort, nx, ny, dx, ox, oy,
                               query_x, query_y, nsamples,
                               periodic, Lx, Ly, seed):
    M = query_x.shape[0]
    out = np.zeros((M, 2), dtype=np.float64)
    area = nx * ny * dx * dx
    pdf = 1.0 / area
    inv_2pi = 1.0 / (2.0 * np.pi)

    for m in prange(M):
        x = query_x[m]
        y = query_y[m]
        sum_u = 0.0
        sum_v = 0.0
        rng_state = np.uint64(seed + m * 1000003)
        for _ in range(nsamples):
            rng_state = np.uint64(rng_state * 6364136223846793005 + 1442695040888963407)
            u1 = np.random.random()
            rng_state = np.uint64(rng_state * 6364136223846793005 + 1442695040888963407)
            u2 = np.random.random()

            yx = ox + u1 * (nx * dx)
            yy = oy + u2 * (ny * dx)

            w = interp_vort_numba(vort, nx, ny, dx, ox, oy, yx, yy, periodic)

            rx = yx - x
            ry = yy - y

            if periodic:
                for ix in range(-1, 2):
                    for iy in range(-1, 2):
                        rx_i = rx + ix * Lx
                        ry_i = ry + iy * Ly
                        r2 = rx_i * rx_i + ry_i * ry_i + 1e-12
                        factor = inv_2pi / r2
                        sum_u += ry_i * factor * (w / pdf)
                        sum_v += -rx_i * factor * (w / pdf)
            else:
                r2 = rx * rx + ry * ry + 1e-12
                factor = inv_2pi / r2
                sum_u += ry * factor * (w / pdf)
                sum_v += -rx * factor * (w / pdf)

        out[m, 0] = sum_u / nsamples
        out[m, 1] = sum_v / nsamples
    return out


from .types import Vec2, RNG
from .grid import Grid

class BiotSavart:
    @staticmethod
    def kernel(x: Vec2, y: Vec2) -> Vec2:
        r = y - x
        r2 = r.norm2() + 1e-12
        return Vec2(r.y / (2 * np.pi * r2), -r.x / (2 * np.pi * r2))

    @staticmethod
    def estimate_velocity(prev_vort: Grid, x: Vec2, nsamples: int,
                          rng: RNG, periodic: bool = False) -> Vec2:
        xs = np.array([x.x])
        ys = np.array([x.y])
        vel = biot_savart_velocity_numba(
            prev_vort.vort, prev_vort.nx, prev_vort.ny, prev_vort.dx,
            prev_vort.ox, prev_vort.oy, xs, ys, nsamples,
            periodic, prev_vort.nx * prev_vort.dx,
            prev_vort.ny * prev_vort.dx, seed=0)
        return Vec2(vel[0, 0], vel[0, 1])