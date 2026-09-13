"""
Optimized performance comparison: 2022 (accelerated fused kernel) vs 2024 (VPL accelerated).
Generates polished comparison figures:
1. Bar chart: per-step time (original vs accelerated, both methods)
2. Curve: error vs samples (2022 vs 2024)
3. Pareto: error vs total time (accelerated)
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU
from gpu_fluid_2022_opt import Sim2022GPUOpt

# ============ Timing: 2022 original vs accelerated ============
def time_2022(cls, res, nmc, steps=50):
    sim = cls(res, res, 0.05, nmc)
    sim._init_taylor_green()
    for _ in range(3): sim.step()
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    for _ in range(steps): sim.step()
    cp.cuda.Stream.null.synchronize()
    return (time.time()-t0)/steps*1000  # ms/step

res = 64; nmc = 256
t22_orig = time_2022(Sim2022GPU, res, nmc)
t22_opt = time_2022(Sim2022GPUOpt, res, nmc)

# 2024 timings (from runs): original 4.96s/20step, accelerated 2.24s/20step
t24_orig = 4.96/20*1000   # ms/step
t24_fast = 2.24/20*1000

print(f"2022: orig={t22_orig:.2f}ms/step, accelerated={t22_opt:.2f}ms/step ({t22_orig/t22_opt:.1f}x)")
print(f"2024: orig={t24_orig:.2f}ms/step, accelerated={t24_fast:.2f}ms/step ({t24_orig/t24_fast:.1f}x)")

# ============ Error data ============
# 2022 errors (center 80%, from earlier)
nmc_list = [64, 256, 1024]
e22 = [0.589, 0.504, 0.537]  # normalized L2
# 2024 errors
samp_list = [256, 512, 1024]
e24 = [0.992, 0.749, 0.609]

# ============ Figure 1: Performance bar chart ============
fig, ax = plt.subplots(figsize=(7, 5))
methods = ['2022\n(original)', '2022\n(accelerated)', '2024\n(original)', '2024\n(accelerated)']
times = [t22_orig, t22_opt, t24_orig, t24_fast]
colors = ['#A8C3E0', '#4C72B0', '#E8A0A0', '#C44E52']
bars = ax.bar(methods, times, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
for b, t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, t*1.02, f'{t:.2f}ms', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Per-step time (ms)', fontsize=12)
ax.set_title('Per-step performance: 2022 vs 2024 (res=64, 256 samples)', fontsize=12)
ax.set_yscale('log')
ax.grid(True, axis='y', alpha=0.3)
# speedup annotations
ax.annotate(f'{t22_orig/t22_opt:.1f}x speedup', xy=(0.5, t22_opt*3), fontsize=10, color='#4C72B0', ha='center')
ax.annotate(f'{t24_orig/t24_fast:.1f}x speedup', xy=(2.5, t24_fast*3), fontsize=10, color='#C44E52', ha='center')
plt.tight_layout()
plt.savefig('report_results/fig_perf_bar.png', dpi=150)
plt.close()

# ============ Figure 2: Error vs samples curves ============
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(nmc_list, e22, 'b-o', label='2022 (accelerated, nmc)', linewidth=2.5, markersize=8)
ax.loglog(samp_list, e24, 'r-s', label='2024 (accelerated, samples)', linewidth=2.5, markersize=8)
ax.set_xlabel('MC samples', fontsize=12)
ax.set_ylabel('Normalized L2 error', fontsize=12)
ax.legend(fontsize=11)
ax.set_title('Error vs MC samples (center 80%, t=1.0)', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('report_results/fig_error_curves.png', dpi=150)
plt.close()

# ============ Figure 3: Pareto with accelerated costs ============
# Total time for 20 steps (normalize)
# 2022 accelerated: t22_opt*20/1000 s (per 20 steps)
# 2024 accelerated: t24_fast*20/1000 s
cost22 = t22_opt*20/1000
cost24 = t24_fast*20/1000
fig, ax = plt.subplots(figsize=(7, 5))
# 2022 points (error at different nmc, cost scales ~linearly with nmc)
for i, (nmc, e) in enumerate(zip(nmc_list, e22)):
    ax.loglog([cost22*nmc/nmc_list[1]], [e], 'bo', markersize=9)
ax.loglog([cost22*nmc/nmc_list[1] for nmc in nmc_list], e22, 'b-', label='2022 (accelerated)', linewidth=2)
# 2024 points
for i, (s, e) in enumerate(zip(samp_list, e24)):
    ax.loglog([cost24*s/samp_list[0]], [e], 'rs', markersize=9)
ax.loglog([cost24*s/samp_list[0] for s in samp_list], e24, 'r-', label='2024 (accelerated)', linewidth=2)
ax.set_xlabel('Total time (s, 20 steps, res=64)', fontsize=12)
ax.set_ylabel('Normalized L2 error', fontsize=12)
ax.legend(fontsize=11)
ax.set_title('Error vs cost Pareto (accelerated)', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('report_results/fig_pareto_accel.png', dpi=150)
plt.close()

print("Saved fig_perf_bar.png, fig_error_curves.png, fig_pareto_accel.png")
print(f"\n=== Accelerated comparison (res=64, 256 samples) ===")
print(f"2022 accelerated: {t22_opt:.2f}ms/step, L2={e22[1]:.3f}")
print(f"2024 accelerated: {t24_fast:.2f}ms/step, L2={e24[0]:.3f}")
print(f"Speed ratio: {t24_fast/t22_opt:.1f}x (2022 faster)")
