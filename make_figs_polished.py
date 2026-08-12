"""
Regenerate all report figures with unified professional styling.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ Unified style ============
C22 = '#1f77b4'   # 2022 blue
C24 = '#d62728'   # 2024 red
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12.5,
    'axes.labelsize': 11.5,
    'axes.titleweight': 'bold',
    'axes.edgecolor': '#666666',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.edgecolor': '#999999',
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})
os.chdir('C:\\Users\\10772\\Desktop\\蒙特卡洛流体仿真')

# ============ Data ============
nmc_list = [64, 256, 1024]
e22 = [0.589, 0.504, 0.537]
e22_linf = [2.14, 2.01, 2.07]
samp_list = [256, 512, 1024]
e24 = [0.992, 0.749, 0.609]
e24_linf = [4.32, 3.57, 2.85]

# ============ Fig: error vs samples (polished) ============
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(nmc_list, e22, 'o-', color=C22, lw=2.5, ms=9, mfc='white', mec=C22, mew=2,
          label='2022 method (nmc)', zorder=3)
ax.loglog(samp_list, e24, 's-', color=C24, lw=2.5, ms=9, mfc='white', mec=C24, mew=2,
          label='2024 method (samples)', zorder=3)
# Annotations
for x, y in zip(nmc_list, e22):
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, -16), ha='center', fontsize=9, color=C22)
for x, y in zip(samp_list, e24):
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9, color=C24)
ax.axhline(0.504, color=C22, ls=':', lw=1.2, alpha=0.7)
ax.text(150, 0.505, '2022 plateau', color=C22, fontsize=9, va='bottom')
ax.set_xlabel('MC samples', fontsize=12)
ax.set_ylabel('Normalized L2 error', fontsize=12)
ax.set_title('Error convergence vs MC samples', fontsize=13, pad=10)
ax.legend(loc='upper right')
ax.set_ylim(0.4, 1.3)
plt.savefig('report_results/fig_error_curves.png')
plt.close()
print('fig_error_curves done')

# ============ Fig: performance bar (polished) ============
t22_orig, t22_opt = 1.67, 0.33
t24_orig, t24_fast = 248.0, 112.0
fig, ax = plt.subplots(figsize=(7.5, 5))
labels = ['2022\noriginal', '2022\naccelerated', '2024\noriginal', '2024\naccelerated']
times = [t22_orig, t22_opt, t24_orig, t24_fast]
colors = ['#9ecae1', C22, '#fcbba1', C24]
bars = ax.bar(labels, times, color=colors, width=0.6, edgecolor='black', linewidth=0.8, zorder=3)
for b, t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, t*1.08, f'{t:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
# speedup annotations
ax.annotate(f'{t22_orig/t22_opt:.1f}x\nfaster', xy=(0.5, t22_opt*2.5), ha='center', fontsize=10, color=C22, fontweight='bold')
ax.annotate(f'{t24_orig/t24_fast:.1f}x\nfaster', xy=(2.5, t24_fast*2.2), ha='center', fontsize=10, color=C24, fontweight='bold')
ax.set_yscale('log')
ax.set_ylabel('Per-step time (ms, log scale)', fontsize=12)
ax.set_title('GPU performance: 2022 vs 2024\n(res=64, 256 samples)', fontsize=13, pad=10)
ax.set_ylim(0.1, 1000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('report_results/fig_perf_bar.png')
plt.close()
print('fig_perf_bar done')

# ============ Fig: Pareto (polished) ============
cost22 = t22_opt*20/1000
cost24 = t24_fast*20/1000
fig, ax = plt.subplots(figsize=(7, 5))
for i, (nmc, e) in enumerate(zip(nmc_list, e22)):
    ax.loglog([cost22*nmc/nmc_list[1]], [e], 'o', color=C22, ms=9, mfc='white', mec=C22, mew=2, zorder=3)
    ax.annotate(f'nmc={nmc}', (cost22*nmc/nmc_list[1], e), textcoords='offset points',
                xytext=(8, 4), fontsize=9, color=C22)
ax.loglog([cost22*nmc/nmc_list[1] for nmc in nmc_list], e22, '-', color=C22, lw=2, label='2022 method')
for i, (s, e) in enumerate(zip(samp_list, e24)):
    ax.loglog([cost24*s/samp_list[0]], [e], 's', color=C24, ms=9, mfc='white', mec=C24, mew=2, zorder=3)
    ax.annotate(f'{s}', (cost24*s/samp_list[0], e), textcoords='offset points',
                xytext=(8, -12), fontsize=9, color=C24)
ax.loglog([cost24*s/samp_list[0] for s in samp_list], e24, '-', color=C24, lw=2, label='2024 method')
ax.set_xlabel('Total time for 20 steps (s)', fontsize=12)
ax.set_ylabel('Normalized L2 error', fontsize=12)
ax.set_title('Error-cost Pareto trade-off\n(accelerated)', fontsize=13, pad=10)
ax.legend(loc='upper right')
ax.set_xlim(1e-3, 10)
ax.set_ylim(0.4, 1.3)
plt.savefig('report_results/fig_pareto_accel.png')
plt.close()
print('fig_pareto_accel done')

# ============ Fig: comprehensive metrics (polished) ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
# Panel 1: metric comparison
metrics = ['L2', 'KE err', 'Max w err']
v22 = [0.504, 0.278, 0.019]
v24 = [0.992, 0.959, 0.949]
x = np.arange(3); w = 0.32
b1 = axes[0].bar(x-w/2, v22, w, color=C22, edgecolor='black', lw=0.5, label='2022', zorder=3)
b2 = axes[0].bar(x+w/2, v24, w, color=C24, edgecolor='black', lw=0.5, label='2024', zorder=3)
for xi, (a, b) in enumerate(zip(v22, v24)):
    axes[0].text(xi-w/2, a+0.02, f'{a:.3f}', ha='center', va='bottom', fontsize=8.5, color=C22)
    axes[0].text(xi+w/2, b+0.02, f'{b:.3f}', ha='center', va='bottom', fontsize=8.5, color=C24)
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics)
axes[0].set_ylabel('Error')
axes[0].set_title('Error metrics (256 samples)')
axes[0].legend()
# Panel 2: sample convergence
axes[1].loglog(samp_list, e24, 's-', color=C24, lw=2.5, ms=8, mfc='white', mec=C24, mew=2, label='2024')
axes[1].axhline(0.504, color=C22, ls='--', lw=2, label='2022 (nmc=256)')
axes[1].set_xlabel('MC samples'); axes[1].set_ylabel('Normalized L2')
axes[1].set_title('2024 sample convergence')
axes[1].legend(loc='upper right')
# Panel 3: resolution convergence
res_list = [32, 64, 128]
errs_res = [0.607, 0.547, 0.493]
axes[2].semilogy(res_list, errs_res, 'o-', color=C22, lw=2.5, ms=8, mfc='white', mec=C22, mew=2)
axes[2].set_xlabel('Grid resolution'); axes[2].set_ylabel('RMSE')
axes[2].set_title('2022 resolution convergence')
axes[2].set_xticks(res_list)
plt.tight_layout()
plt.savefig('report_results/fig_comprehensive_metrics.png')
plt.close()
print('fig_comprehensive_metrics done')

# ============ Fig: error evolution + KE + divergence (polished) ============
import sys
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

def read_vel(path):
    with open(path,'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1,2)
def curl(u,v,h,res):
    w=np.zeros((res,res)); w[1:-1,1:-1]=(v[1:-1,2:]-v[1:-1,:-2])/(2*h)-(u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w

# 2022 error evolution
res = 64
sim = Sim2022GPU(res, res, 0.05, 256, nu=0.05)
sim._init_taylor_green()
xs2 = (np.arange(res)+0.5)*sim.dx + sim.ox
X2, Y2 = np.meshgrid(xs2, xs2, indexing='ij')
w02 = 2*np.pi*np.cos(np.pi*X2)*np.cos(np.pi*Y2)
err22_t, ke22_t = [], []
for s in range(20):
    sim.step()
    we = w02*np.exp(-2*np.pi**2*0.05*(s+1)*0.05)
    err22_t.append(np.sqrt(np.mean((sim.vorticity-we)**2)))
    ke22_t.append(0.5*np.sum(sim.vorticity**2))

# 2024 error evolution
L=2.4; h=L/res
xs=np.linspace(-L/2+h/2,L/2-h/2,res)
X,Y=np.meshgrid(xs,xs,indexing='xy')
k=2*np.pi/L
err24_t, ke24_t = [], []
for s in range(1, 21):
    vel=read_vel(f'VelMCFluids/results/results_tg_compare/raw/velocity_{s}.vector')
    u=vel[:,0].reshape(res,res); v=vel[:,1].reshape(res,res)
    w=curl(u,v,h,res)
    we=2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*s*0.05)
    err24_t.append(np.sqrt(np.mean((w-we)**2)))
    ke24_t.append(0.5*np.sum(vel[:,0]**2+vel[:,1]**2))

t=(np.arange(20)+1)*0.05
ke22_ana = np.array([0.5*np.sum((w02*np.exp(-2*np.pi**2*0.05*s*0.05))**2) for s in range(21)])
ke24_ana = np.array([0.5*np.sum((2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*s*0.05))**2) for s in range(21)])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].semilogy(t, err22_t, 'o-', color=C22, lw=2, ms=5, mfc='white', mec=C22, mew=1.5, label='2022')
axes[0].semilogy(t, err24_t, 's-', color=C24, lw=2, ms=5, mfc='white', mec=C24, mew=1.5, label='2024')
axes[0].set_xlabel('Time t'); axes[0].set_ylabel('Vorticity RMSE')
axes[0].set_title('Error evolution (256 samples)')
axes[0].legend()
axes[1].plot(t, ke22_t/ke22_t[0], 'o-', color=C22, lw=2, ms=4, label='2022', mfc='white')
axes[1].plot(t, ke24_t/ke24_t[0], 's-', color=C24, lw=2, ms=4, label='2024', mfc='white')
axes[1].plot(t, ke22_ana[1:]/ke22_ana[0], '--', color=C22, lw=1.5, alpha=0.6, label='2022 analytic')
axes[1].plot(t, ke24_ana[1:]/ke24_ana[0], '--', color=C24, lw=1.5, alpha=0.6, label='2024 analytic')
axes[1].set_xlabel('Time t'); axes[1].set_ylabel('KE / KE(0)')
axes[1].set_title('Kinetic energy decay (viscous)')
axes[1].legend(fontsize=9)
div24=[]
for s in range(1,21):
    vel=read_vel(f'VelMCFluids/results/results_tg_compare/raw/velocity_{s}.vector')
    u=vel[:,0].reshape(res,res); v=vel[:,1].reshape(res,res)
    d=(u[1:-1,2:]-u[1:-1,:-2])/(2*h)+(v[2:,1:-1]-v[:-2,1:-1])/(2*h)
    div24.append(np.mean(np.abs(d)))
axes[2].plot(t, div24, 's-', color=C24, lw=2, ms=5, mfc='white', mec=C24, mew=1.5)
axes[2].axhline(0, color='k', ls='--', lw=1)
axes[2].set_xlabel('Time t'); axes[2].set_ylabel('Mean |div u|')
axes[2].set_title('2024 divergence error')
axes[2].set_yscale('log')
plt.tight_layout()
plt.savefig('report_results/fig_error_ke.png')
plt.close()
print('fig_error_ke done')

print('All polished figures saved.')
