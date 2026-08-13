"""
Extended sample count comparison: do higher samples narrow the 2022 vs 2024 gap?
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

def read_vel(path):
    with open(path,'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1,2)
def curl(u,v,h,res):
    w=np.zeros((res,res)); w[1:-1,1:-1]=(v[1:-1,2:]-v[1:-1,:-2])/(2*h)-(u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w
def mask_center(w, frac=0.8):
    m=int((1-frac)/2*w.shape[0])
    return w[m:-m,m:-m]

# 2022 errors at high nmc
nmc_list = [256, 1024, 4096, 16384]
e22 = []
for nmc in nmc_list:
    sim = Sim2022GPU(64, 64, 0.05, nmc, nu=0.05)
    sim._init_taylor_green()
    xs=(np.arange(64)+0.5)*sim.dx+sim.ox
    X,Y=np.meshgrid(xs,xs,indexing='ij')
    w0=2*np.pi*np.cos(np.pi*X)*np.cos(np.pi*Y)
    for s in range(20): sim.step()
    we=w0*np.exp(-2*np.pi**2*0.05*1.0)
    wc,wac=mask_center(sim.vorticity),mask_center(we)
    e22.append(np.linalg.norm(wc-wac)/np.linalg.norm(wac))
    print(f"2022 nmc={nmc}: L2={e22[-1]:.4f}")

# 2024 errors at high samples
res=64; L=2.4; h=L/res
xs=np.linspace(-L/2+h/2,L/2-h/2,res)
X,Y=np.meshgrid(xs,xs,indexing='xy')
k=2*np.pi/L
we24=2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*1.0)
samp_list=[256,512,1024,2048,4096]
dirs={256:'results_tg_compare',512:'results_tg_s512',1024:'results_tg_s1024',
      2048:'results_tg_hi2048',4096:'results_tg_hi4096'}
e24=[]
for s in samp_list:
    vel=read_vel(f'VelMCFluids/results/{dirs[s]}/raw/velocity_20.vector')
    u=vel[:,0].reshape(res,res); v=vel[:,1].reshape(res,res)
    w=curl(u,v,h,res)
    wc,wac=mask_center(w),mask_center(we24)
    e24.append(np.linalg.norm(wc-wac)/np.linalg.norm(wac))
    print(f"2024 samples={s}: L2={e24[-1]:.4f}")

np.savez('report_results/high_samples.npz', nmc_list=nmc_list, e22=e22,
         samp_list=samp_list, e24=e24)

# Plot extended convergence
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.loglog(nmc_list, e22, 'o-', color='#1f77b4', lw=2.5, ms=9, mfc='white', mec='#1f77b4', mew=2,
          label='2022 (nmc)', zorder=3)
ax.loglog(samp_list, e24, 's-', color='#d62728', lw=2.5, ms=9, mfc='white', mec='#d62728', mew=2,
          label='2024 (samples)', zorder=3)
for x,y in zip(nmc_list,e22):
    ax.annotate(f'{y:.3f}',(x,y),textcoords='offset points',xytext=(0,-16),ha='center',fontsize=9,color='#1f77b4')
for x,y in zip(samp_list,e24):
    ax.annotate(f'{y:.3f}',(x,y),textcoords='offset points',xytext=(0,8),ha='center',fontsize=9,color='#d62728')
ax.set_xlabel('MC samples', fontsize=12)
ax.set_ylabel('Normalized L2 error', fontsize=12)
ax.set_title('Error convergence with extended sample counts\n(center 80%, t=1.0)', fontsize=12.5)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, which='both', alpha=0.25, ls='--')
ax.set_ylim(0.4, 1.2)
plt.savefig('report_results/fig_high_samples.png')
plt.close()

print("\n=== 扩展采样结论 ===")
print(f"2022: nmc 256->16384: L2 {e22[0]:.3f}->{e22[-1]:.3f} (平台)")
print(f"2024: samples 256->4096: L2 {e24[0]:.3f}->{e24[-1]:.3f} (持续下降)")
print(f"4096采样时差距: 2022 {e22[-2]:.3f} vs 2024 {e24[-1]:.3f}")
print("Saved fig_high_samples.png")
