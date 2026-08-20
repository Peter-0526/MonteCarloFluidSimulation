import numpy as np
from numba import njit, prange
from .grid import Grid
from .biot_savart import biot_savart_velocity_numba, interp_vort_numba
from .types import Vec2, RNG


@njit(parallel=True, cache=True)
def update_vorticity_numba(vort_prev, vel_prev, vort_next,
                           nx, ny, dx, ox, oy, dt, nu, nd,
                           periodic, Lx, Ly, seed):
    scale_diff = np.sqrt(2.0 * nu * dt)
    for idx in prange(nx * ny):
        i = idx % nx
        j = idx // nx
        x = ox + (i + 0.5) * dx
        y = oy + (j + 0.5) * dx

        u = vel_prev[idx, 0]
        v = vel_prev[idx, 1]

        xb = x - u * dt
        yb = y - v * dt
        if periodic:
            xb = ox + ((xb - ox) % Lx)
            yb = oy + ((yb - oy) % Ly)

        acc = 0.0
        rng_state = np.uint64(seed + idx * 1000003)
        for _ in range(nd):
            g1 = np.random.randn()
            g2 = np.random.randn()

            xd = xb + scale_diff * g1
            yd = yb + scale_diff * g2
            if periodic:
                xd = ox + ((xd - ox) % Lx)
                yd = oy + ((yd - oy) % Ly)

            acc += interp_vort_numba(vort_prev, nx, ny, dx, ox, oy, xd, yd, periodic)

        vort_next[j, i] = acc / nd


class Simulator:
    def __init__(self, nx, ny, dt, nu=0.0, L=1.0, periodic=False,
                 nmc=256, nd=4):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.nu = nu
        self.L = L
        self.dx = 2.0 * L / nx
        self.ox, self.oy = -L, -L
        self.grid = [Grid(nx, ny, self.dx, self.ox, self.oy) for _ in range(2)]
        self.cur = 0
        self.nmc = nmc
        self.nd = nd
        self.periodic = periodic
        self.rng = RNG()
        self._init_vorticity()

    def _init_vorticity(self):
        sigma = 0.2
        centers = [Vec2(-0.5, 0.0), Vec2(0.5, 0.0)]
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.grid[0].grid_pos(i, j)
                w = 0.0
                for c in centers:
                    d = p - c
                    w += np.exp(-d.norm2() / (2 * sigma * sigma))
                self.grid[0].set_vort(i, j, w)
                self.grid[1].set_vort(i, j, w)

    def set_vorticity_field(self, func):
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.grid[0].grid_pos(i, j)
                val = func(p)
                self.grid[0].set_vort(i, j, val)
                self.grid[1].set_vort(i, j, val)

    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1 - self.cur]
        nx, ny = self.nx, self.ny
        dx = self.dx
        ox, oy = self.ox, self.oy

        xs = np.zeros(nx * ny)
        ys = np.zeros(nx * ny)
        idx = 0
        for j in range(ny):
            for i in range(nx):
                p = prev.grid_pos(i, j)
                xs[idx] = p.x
                ys[idx] = p.y
                idx += 1

        vel = biot_savart_velocity_numba(
            prev.vort, nx, ny, dx, ox, oy, xs, ys, self.nmc,
            self.periodic, nx * dx, ny * dx, seed=12345)

        update_vorticity_numba(
            prev.vort, vel, nxt.vort, nx, ny, dx, ox, oy,
            self.dt, self.nu, self.nd, self.periodic,
            nx * dx, ny * dx, seed=67890)

        self.cur = 1 - self.cur

    @property
    def vorticity(self):
        return self.grid[self.cur].vort

    def reset(self):
        self.cur = 0
        self._init_vorticity()