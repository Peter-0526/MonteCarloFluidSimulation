"""
Optimized GPU 2022 vorticity method.
Scheme A: Custom fused CUDA kernel for Biot-Savart (RawKernel).
Scheme B: Fused bilinear interpolation kernel.
"""
import numpy as np
import cupy as cp
import time

# ============ Scheme A: fused Biot-Savart kernel ============
BIOT_KERNEL = r'''
extern "C" __global__
void biot_savart_fused(const float2* query, const float* vort, const float2* samples,
                       const int nx, const int ny, const float ox, const float oy, const float dx,
                       const int N, const int P, float2* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    const float qx = query[i].x, qy = query[i].y;
    const float inv_2pi = 0.15915494f;
    const float inv_pdf = 4.0f; // 1/pdf, pdf=1/4
    float sum_x = 0.f, sum_y = 0.f;

    for (int k = 0; k < N; k++) {
        const float sx = samples[k].x, sy = samples[k].y;
        // bilinear interpolate vorticity at sample point
        float fx = (sx - ox) / dx - 0.5f;
        float fy = (sy - oy) / dx - 0.5f;
        int i0 = (int)floorf(fx), j0 = (int)floorf(fy);
        float tx = fx - i0, ty = fy - j0;
        i0 = i0 < 0 ? 0 : (i0 > nx-2 ? nx-2 : i0);
        j0 = j0 < 0 ? 0 : (j0 > ny-2 ? ny-2 : j0);
        int i1 = i0+1, j1 = j0+1;
        float v00 = vort[j0*nx+i0], v10 = vort[j0*nx+i1];
        float v01 = vort[j1*nx+i0], v11 = vort[j1*nx+i1];
        float w = ((1-ty)*((1-tx)*v00 + tx*v10) + ty*((1-tx)*v01 + tx*v11));

        float rx = qx - sx, ry = qy - sy;
        float r2 = rx*rx + ry*ry + 1e-12f;
        sum_x += (ry * inv_2pi / r2) * (w * inv_pdf);
        sum_y += (-rx * inv_2pi / r2) * (w * inv_pdf);
    }
    out[i].x = sum_x / N;
    out[i].y = sum_y / N;
}
'''

# ============ Scheme B: fused bilinear interpolation kernel ============
BILIN_KERNEL = r'''
extern "C" __global__
void bilinear_fused(const float2* pos, const float* vort,
                    const int nx, const int ny, const float ox, const float oy, const float dx,
                    const int P, float* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    const float x = pos[i].x, y = pos[i].y;
    float fx = (x - ox) / dx - 0.5f;
    float fy = (y - oy) / dx - 0.5f;
    int i0 = (int)floorf(fx), j0 = (int)floorf(fy);
    float tx = fx - i0, ty = fy - j0;
    i0 = i0 < 0 ? 0 : (i0 > nx-2 ? nx-2 : i0);
    j0 = j0 < 0 ? 0 : (j0 > ny-2 ? ny-2 : j0);
    int i1 = i0+1, j1 = j0+1;
    float v00 = vort[j0*nx+i0], v10 = vort[j0*nx+i1];
    float v01 = vort[j1*nx+i0], v11 = vort[j1*nx+i1];
    out[i] = ((1-ty)*((1-tx)*v00 + tx*v10) + ty*((1-tx)*v01 + tx*v11));
}
'''

class Sim2022GPUOpt:
    """Optimized GPU 2022 with fused kernels."""
    def __init__(self, nx, ny, dt, nmc, nu=0.0):
        self.nx, self.ny, self.dt, self.nmc, self.nu = nx, ny, dt, nmc, nu
        self.dx = 2.0/nx
        self.ox, self.oy = -1.0, -1.0
        self.vort = cp.zeros((ny, nx), dtype=cp.float32)
        rng = np.random.default_rng(42)
        self.samples = cp.asarray(rng.uniform(-1, 1, (nmc, 2)), dtype=cp.float32)
        # compile kernels
        self.bs_mod = cp.RawModule(code=BIOT_KERNEL)
        self.bs_kernel = self.bs_mod.get_function('biot_savart_fused')
        self.bi_mod = cp.RawModule(code=BILIN_KERNEL)
        self.bi_kernel = self.bi_mod.get_function('bilinear_fused')
        # query positions (P,2) as float2
        xs = (np.arange(nx)+0.5)*self.dx + self.ox
        ys = (np.arange(ny)+0.5)*self.dx + self.oy
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        qpos = np.stack([XX, YY], axis=-1).reshape(-1, 2).astype(np.float32)
        self.qpos = cp.asarray(qpos)
        self.qpos2 = cp.asarray(qpos.view(np.float32).reshape(-1, 2))  # as float2 via view
        self.P = self.nx*self.ny
        self.samples2 = self.samples.view(np.float32).reshape(-1, 2)
        self._init_taylor_green()

    def _init_taylor_green(self):
        for j in range(self.ny):
            for i in range(self.nx):
                px = self.ox + (i+0.5)*self.dx
                py = self.oy + (j+0.5)*self.dx
                self.vort[j, i] = 2*np.pi*np.cos(np.pi*px)*np.cos(np.pi*py)

    def biot_savart_fused(self):
        out = cp.empty((self.P, 2), dtype=cp.float32)
        block = 256
        grid = (self.P + block - 1)//block
        self.bs_kernel((grid,), (block,), (self.qpos2, self.vort, self.samples2,
                       np.int32(self.nx), np.int32(self.ny), np.float32(self.ox), np.float32(self.oy),
                       np.float32(self.dx), np.int32(self.nmc), np.int32(self.P), out))
        return out

    def interpolate(self, pos):
        """pos: (P,2) float32 numpy/cupy. Returns (P,)"""
        if isinstance(pos, np.ndarray):
            pos2 = cp.asarray(pos.reshape(-1,2))
        else:
            pos2 = pos.reshape(-1, 2)
        out = cp.empty(pos2.shape[0], dtype=cp.float32)
        P = pos2.shape[0]
        block = 256
        grid = (P + block - 1)//block
        self.bi_kernel((grid,), (block,), (pos2, self.vort,
                       np.int32(self.nx), np.int32(self.ny), np.float32(self.ox), np.float32(self.oy),
                       np.float32(self.dx), np.int32(P), out))
        return out

    def step(self):
        # Biot-Savart velocity (fused kernel)
        vel = self.biot_savart_fused()
        # Semi-Lagrangian backtrace
        back_x = cp.clip(self.qpos2[:,0] - vel[:,0]*self.dt, -1, 1)
        back_y = cp.clip(self.qpos2[:,1] - vel[:,1]*self.dt, -1, 1)
        back = cp.stack([back_x, back_y], axis=-1)
        w_new = self.interpolate(back)
        # diffusion
        if self.nu > 0:
            sigma = np.sqrt(2*self.nu*self.dt)
            rng = cp.random.default_rng()
            Z = rng.standard_normal((self.P, 2), dtype=cp.float32)*sigma
            diff = cp.clip(back + Z, -1, 1)
            w_new = self.interpolate(diff)
        self.vort.reshape(-1)[...] = w_new

    @property
    def vorticity(self):
        return cp.asnumpy(self.vort)


if __name__ == '__main__':
    import sys
    from gpu_fluid_2022 import Sim2022GPU
    for res in [64, 128, 256]:
        nmc = 256
        # Benchmark original vs optimized
        sim0 = Sim2022GPU(res, res, 0.05, nmc)
        sim0._init_taylor_green()
        t0 = time.time()
        for _ in range(10): sim0.step()
        cp.cuda.Stream.null.synchronize()
        t_orig = (time.time()-t0)/10*1000

        sim1 = Sim2022GPUOpt(res, res, 0.05, nmc)
        t0 = time.time()
        for _ in range(10): sim1.step()
        cp.cuda.Stream.null.synchronize()
        t_opt = (time.time()-t0)/10*1000
        print(f'res={res}: 原版={t_orig:.2f}ms/步, 优化版={t_opt:.2f}ms/步, 加速={t_orig/t_opt:.1f}x')
