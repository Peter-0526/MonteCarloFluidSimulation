"""
2022 vorticity method with the SAME initial condition as 2024 scene 3 (vortex pair).
Two opposite-sign vortices at (-0.7, 1/6) and (-0.7, -1/6), radius 0.8/6.
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.types import Vec2, RNG as CoreRNG
from core.grid import Grid
from core.biot_savart import BiotSavart
import random as pyrandom

class RNG(CoreRNG):
    def __init__(self):
        self.uni = pyrandom.uniform

    def uniform(self, a, b):
        return self.uni(a, b)

    def uniform_in_box(self, xmin, xmax, ymin, ymax):
        return Vec2(self.uniform(xmin, xmax), self.uniform(ymin, ymax))

class Sim2022:
    def __init__(self, nx, ny, dt, nmc):
        self.nx, self.ny, self.dt, self.nmc = nx, ny, dt, nmc
        self.dx = 2.0 / nx
        self.grid = [Grid(nx, ny, self.dx, -1.0, -1.0) for _ in range(2)]
        self.cur = 0
        self.rng = RNG()
        # 2024 scene 3 vortex pair: opposite sign, radius 0.8/6, centers (-0.7, +/-1/6)
        c1 = Vec2(-0.7, 1.0/6.0)
        c2 = Vec2(-0.7, -1.0/6.0)
        radius = 0.8 / 6.0
        for j in range(ny):
            for i in range(nx):
                p = self.grid[0].grid_pos(i, j)
                w = 0.0
                d1 = p - c1
                if d1.norm() <= radius: w += 1.0
                d2 = p - c2
                if d2.norm() <= radius: w -= 1.0
                self.grid[0].set_vort(i, j, w)
                self.grid[1].set_vort(i, j, w)
        self.frames = [self.vorticity.copy()]
    def step(self):
        prev, nxt = self.grid[self.cur], self.grid[1-self.cur]
        for j in range(self.ny):
            for i in range(self.nx):
                x = prev.grid_pos(i, j)
                v = BiotSavart.estimate_velocity(prev, x, self.nmc, self.rng)
                xb = x - v*self.dt
                w = prev.get_vort(xb)
                nxt.set_vort(i, j, w)
        self.cur = 1 - self.cur
        self.frames.append(self.vorticity.copy())
    @property
    def vorticity(self): return self.grid[self.cur].vort

# Run
res, dt, nmc, steps = 64, 0.05, 128, 40  # match 2024 t=2
sim = Sim2022(res, res, dt, nmc)
t0 = time.time()
for _ in range(steps):
    sim.step()
t_elapsed = time.time() - t0
print(f"2022 vortex-pair: {res}x{res}, {steps} steps, time={t_elapsed:.1f}s")

# Save final frame + animation frames
os.makedirs('compare_output', exist_ok=True)
vmax = 1.0
# Plot selected frames
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for idx, fstep in enumerate([0, 10, 20, 30, 39]):
    ax = axes.flat[idx]
    ax.imshow(sim.frames[fstep].T, origin='lower', cmap='RdBu_r',
              vmin=-vmax, vmax=vmax, extent=[-1,1,-1,1])
    ax.set_title(f't={fstep*dt:.1f}')
axes.flat[5].axis('off')
plt.tight_layout()
plt.savefig('compare_output/2022_vortex_pair_frames.png', dpi=120)
plt.close()
print("Saved compare_output/2022_vortex_pair_frames.png")
