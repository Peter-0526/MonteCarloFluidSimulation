"""
Apply collaborator's methodology to the GPU comparison:
1. Normalized L2 + L-infinity dual metrics
2. Center-region masking (80%)
3. Repeat trials for variance quantification
4. Pareto error-cost trade-off
"""
import sys, os
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

def read_vel(path):
    with open(path, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 2)
def curl(u, v, h, res):
    w = np.zeros((res, res))
    w[1:-1, 1:-1] = (v[1:-1,2:]-v[1:-1,:-2])/(2*h) - (u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w

def mask_center(w, frac=0.8):
    """Center-region mask to exclude boundary artifacts."""
    res = w.shape[0]
    m = int((1-frac)/2*res)
    return w[m:res-m, m:res-m]

# ============ 2022 with repeats (mean±std) ============
def run_2022_once(res, nmc, steps=20, nu=0.05):
    sim = Sim2022GPU(res, res, 0.05, nmc, nu=nu)
    sim._init_taylor_green()
    xs = (np.arange(res)+0.5)*sim.dx + sim.ox
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    w0 = 2*np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)
    for s in range(steps): sim.step()
    w = sim.vorticity
    t = steps*0.05
    wa = w0*np.exp(-2*np.pi**2*nu*t)
    # L2 relative + L-infinity, center 80%
    wc, wac = mask_center(w), mask_center(wa)
    l2 = np.linalg.norm(wc-wac)/np.linalg.norm(wac)
    linf = np.max(np.abs(wc-wac))
    return l2, linf

nmc_list = [64, 256, 1024]
print("=== 2022 GPU: NMC convergence with repeats (mean±std) ===")
print("nmc\tmean_L2\tstd_L2\tmean_Linf")
l22_l2, l22_linf = [], []
t22_cost = []
for nmc in nmc_list:
    l2s, linfs = [], []
    t0 = time.time()
    for r in range(3):  # 3 repeats
        l2, linf = run_2022_once(64, nmc)
        l2s.append(l2); linfs.append(linf)
    t22_cost.append(time.time()-t0)
    l22_l2.append(np.mean(l2s)); l22_linf.append(np.mean(linfs))
    print(f"{nmc}\t{np.mean(l2s):.6f}\t{np.std(l2s):.6f}\t{np.mean(linfs):.4f}")

# ============ 2024 (from saved output, single run) ============
res = 64; L = 2.4; h = L/res
xs = np.linspace(-L/2+h/2, L/2-h/2, res)
X, Y = np.meshgrid(xs, xs, indexing='xy')
k = 2*np.pi/L; t = 1.0
wa24 = 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*t)

samp_list = [256, 512, 1024]
dirs = {256:'results_tg_compare', 512:'results_tg_s512', 1024:'results_tg_s1024'}
l24_l2, l24_linf, t24_cost = [], [], []
for s in samp_list:
    vel = read_vel(f'VelMCFluids/results/{dirs[s]}/raw/velocity_20.vector')
    u = vel[:,0].reshape(res,res); v = vel[:,1].reshape(res,res)
    w = curl(u, v, h, res)
    wc, wac = mask_center(w), mask_center(wa24)
    l24_l2.append(np.linalg.norm(wc-wac)/np.linalg.norm(wac))
    l24_linf.append(np.max(np.abs(wc-wac)))

print("\n=== 2024 GPU: NMC convergence (center 80%) ===")
print("samples\tL2\tLinf")
for s, l2, li in zip(samp_list, l24_l2, l24_linf):
    print(f"{s}\t{l2:.6f}\t{li:.4f}")

# ============ Pareto figure ============
# Timing (per run): 2022 measured above; 2024 from earlier runs
t24_s = [4.96, 4.27, 2.04]  # 20 steps each (config_tg_compare and s512/s1024)
# Normalize to per-20-steps
fig, ax = plt.subplots(figsize=(6.5, 5))
# 2022: cost ~ per run (3 repeats amortized /3)
c22 = [t/3 for t in t22_cost]
ax.loglog(c22, l22_l2, 'b-o', label='2022 (nmc)', linewidth=2, markersize=7)
ax.loglog(t24_s, l24_l2, 'r-s', label='2024 (samples)', linewidth=2, markersize=7)
ax.set_xlabel('Compute time (s, 20 steps, res=64)')
ax.set_ylabel('Normalized L2 error (center 80%)')
ax.legend(); ax.set_title('Error vs Cost Pareto')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_results/fig_pareto.png', dpi=150)
plt.close()

np.savez('report_results/borrowed.npz', nmc_list=nmc_list, l22_l2=np.array(l22_l2),
         l22_linf=np.array(l22_linf), samp_list=samp_list,
         l24_l2=np.array(l24_l2), l24_linf=np.array(l24_linf))
print("\nSaved fig_pareto.png")
