"""
Regenerate field comparison + complex boundary figures with polished styling.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.image as mpimg

C22 = '#1f77b4'
C24 = '#d62728'
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11.5,
    'axes.titleweight': 'bold', 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'xtick.direction': 'in', 'ytick.direction': 'in',
})
os.chdir('C:\\Users\\10772\\Desktop\\蒙特卡洛流体仿真')

sys.path.insert(0, '2022_Simulation/mc_fluid_step1_py')
from gpu_fluid_2022 import Sim2022GPU

def read_vel(path):
    with open(path,'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(-1,2)
def curl(u,v,h,res):
    w=np.zeros((res,res)); w[1:-1,1:-1]=(v[1:-1,2:]-v[1:-1,:-2])/(2*h)-(u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    return w

# ============ Field comparison ============
res = 64
sim = Sim2022GPU(res, res, 0.05, 256, nu=0.05)
sim._init_taylor_green()
xs2 = (np.arange(res)+0.5)*sim.dx + sim.ox
X2, Y2 = np.meshgrid(xs2, xs2, indexing='ij')
w02 = 2*np.pi*np.cos(np.pi*X2)*np.cos(np.pi*Y2)
w22_fields = [sim.vorticity.copy()]
for s in range(20): sim.step(); w22_fields.append(sim.vorticity.copy())

L=2.4; h=L/res
xs=np.linspace(-L/2+h/2,L/2-h/2,res)
X,Y=np.meshgrid(xs,xs,indexing='xy')
k=2*np.pi/L
w24_fields=[]
for s in range(21):
    vel=read_vel(f'VelMCFluids/results/results_tg_compare/raw/velocity_{s}.vector')
    u=vel[:,0].reshape(res,res); v=vel[:,1].reshape(res,res)
    w24_fields.append(curl(u,v,h,res))

times = [0, 5, 10, 20]
fig, axes = plt.subplots(3, 4, figsize=(14, 8.5))
row_labels = ['Analytical\n(reference)', '2022 method', '2024 method']
cmap = 'RdBu_r'
for j, s in enumerate(times):
    tt = s*0.05
    wa = w02*np.exp(-2*np.pi**2*0.05*tt)
    wb = 2*k*np.cos(k*X)*np.cos(k*Y)*np.exp(-2*k*k*0.05*tt)
    vmax = max(abs(wa).max(), abs(wb).max())
    for row, data, ext in [(0, wa, [-1,1,-1,1]), (1, w22_fields[s], [-1,1,-1,1]),
                           (2, wb, [-1.2,1.2,-1.2,1.2])]:
        if row == 2:
            data = w24_fields[s]
        axes[row, j].imshow(data.T, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax,
                            extent=ext, aspect='auto')
        axes[row, j].set_title(f't={tt:.2f}' if row == 0 else '', fontsize=10.5)
        axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
        for sp in axes[row, j].spines.values(): sp.set_color('#cccccc'); sp.set_linewidth(0.8)
for row in range(3):
    axes[row, 0].set_ylabel(row_labels[row], fontsize=11, rotation=0, labelpad=30, va='center')
plt.tight_layout()
plt.savefig('report_results/fig_field_comparison.png')
plt.close()
print('fig_field_comparison done')

# ============ Complex boundary comparison (restyle) ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
img22 = mpimg.imread('report_results/fig_2022_boundary_failure.png')
img24 = mpimg.imread('VelMCFluids/results/results_karman_tiny/png/concentration_30.png')
axes[0].imshow(img22)
axes[0].set_title('2022 vorticity method: vortex near circle\n(flow passes THROUGH obstacle, no-penetration violated)',
                  fontsize=11, fontweight='bold', color=C22)
axes[0].axis('off')
axes[1].imshow(img24)
axes[1].set_title('2024 velocity method: flow past cylinder\n(correctly deflects around obstacle, Karman street)',
                  fontsize=11, fontweight='bold', color=C24)
axes[1].axis('off')
for sp in [axes[0].spines, axes[1].spines]:
    for s in sp.values(): s.set_color('#333333'); s.set_linewidth(1.2)
plt.tight_layout()
plt.savefig('report_results/fig_complex_boundary_comparison.png')
plt.close()
print('fig_complex_boundary_comparison done')
