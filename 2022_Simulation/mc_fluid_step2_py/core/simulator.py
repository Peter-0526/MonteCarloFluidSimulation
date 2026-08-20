# core/simulator.py
from .grid import Grid
from .geometry import Geometry
from .wos import WoS_solver
from .types import Vec2, RNG
from .biot_savart import BiotSavart, estimate_velocity_batch   # 导入批量函数
from typing import Optional
import numpy as np

class Simulator:
    def __init__(self, nx, ny, dt, geometry: Optional[Geometry] = None):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = 2.0 / nx
        ox, oy = -1.0, -1.0
        self.grid = [Grid(nx, ny, self.dx, ox, oy) for _ in range(2)]
        self.cur = 0
        self.nmc = 256
        self.paused = False
        self.rng = RNG()
        self.geometry = geometry if geometry is not None else Geometry()
        # 根据是否有障碍物选择速度估计算法
        if len(self.geometry.obstacles) > 0:
            self.wos_solver = WoS_solver(self.geometry, self.grid[0])
            self.use_wos = True
        else:
            self.use_wos = False
        self._init_vorticity()

    def _init_vorticity(self):
        # 在域内初始化两个高斯涡旋
        sigma = 0.2
        centers = [Vec2(-0.5, 0.0), Vec2(0.5, 0.0)]
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.grid[0].grid_pos(i,j)
                # 只初始化域内的点
                if self.geometry.inside_domain(p):
                    w = 0.0
                    for c in centers:
                        d = p - c
                        w += np.exp(-d.norm2() / (2*sigma*sigma))
                else:
                    w = 0.0  # 固体内部不设涡量
                self.grid[0].set_vort(i,j,w)
                self.grid[1].set_vort(i,j,w)

    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1-self.cur]
        # 更新 WoS solver 中的网格引用（使用上一时间步缓存）
        if self.use_wos:
            self.wos_solver.grid = prev

        if self.use_wos:
            # 有障碍物时，沿用原来的逐点方法（WoS_solver 本身可能已优化）
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i,j)
                    if not self.geometry.inside_domain(x):
                        nxt.set_vort(i,j,0.0)
                        continue
                    v = self.wos_solver.velocity_at(x, self.nmc)
                    # 半拉格朗日回溯
                    xb = x - v * self.dt
                    if not self.geometry.inside_domain(xb):
                        xb = self.geometry.closest_point(xb)
                        inward = xb - x
                        if inward.norm() > 1e-8:
                            xb = xb + inward * 1e-4
                    w = prev.get_vort(xb)
                    nxt.set_vort(i,j,w)
        else:
            # 无障碍物时，使用批量并行速度估计
            # 1) 收集所有网格点的坐标（展开成一维数组）
            xs = np.empty(self.nx * self.ny)
            ys = np.empty(self.nx * self.ny)
            idx = 0
            for j in range(self.ny):
                for i in range(self.nx):
                    p = prev.grid_pos(i,j)
                    xs[idx] = p.x
                    ys[idx] = p.y
                    idx += 1
            # 2) 批量估计速度
            vx, vy = estimate_velocity_batch(prev, xs, ys, self.nmc)

            # 3) 逐点进行半拉格朗日回溯（由于没有障碍物，inside_domain 通常恒为 True）
            idx = 0
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i,j)
                    if not self.geometry.inside_domain(x):
                        nxt.set_vort(i,j,0.0)
                        idx += 1
                        continue
                    v = Vec2(vx[idx], vy[idx])
                    idx += 1
                    # 半拉格朗日后向追踪
                    xb = x - v * self.dt
                    # 若回溯点落在域外，则取最近域内点（防止涡量泄漏）
                    if not self.geometry.inside_domain(xb):
                        xb = self.geometry.closest_point(xb)
                        inward = xb - x
                        if inward.norm() > 1e-8:
                            xb = xb + inward * 1e-4
                    w = prev.get_vort(xb)
                    nxt.set_vort(i,j,w)
        self.cur = 1 - self.cur

    @property
    def vorticity(self):
        return self.grid[self.cur].vort

    def reset(self):
        self.cur = 0
        self._init_vorticity()