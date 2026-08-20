import numpy as np
from numba import njit
from .types3d import Vec3


# Numba 单点三线性插值（支持周期边界）
@njit
def trilinear_interp_numba(data, x, y, z, nx, ny, nz, dx, ox, oy, oz, periodic=False):
    """对 data (nz,ny,nx,3) 在 (x,y,z) 处做三线性插值，返回 (3,) 数组"""
    fx = (x - ox) / dx - 0.5
    fy = (y - oy) / dx - 0.5
    fz = (z - oz) / dx - 0.5

    if periodic:
        # 周期索引：将坐标映射到 [0, nx) 区间
        fx = fx % nx
        fy = fy % ny
        fz = fz % nz
        i0 = int(fx);  j0 = int(fy);  k0 = int(fz)
        i1 = (i0 + 1) % nx
        j1 = (j0 + 1) % ny
        k1 = (k0 + 1) % nz
    else:
        i0 = int(fx);  j0 = int(fy);  k0 = int(fz)
        i0 = max(0, min(i0, nx-2)); i1 = i0+1
        j0 = max(0, min(j0, ny-2)); j1 = j0+1
        k0 = max(0, min(k0, nz-2)); k1 = k0+1

    tx = fx - i0; ty = fy - j0; tz = fz - k0

    v000 = data[k0, j0, i0]; v100 = data[k0, j0, i1]
    v010 = data[k0, j1, i0]; v110 = data[k0, j1, i1]
    v001 = data[k1, j0, i0]; v101 = data[k1, j0, i1]
    v011 = data[k1, j1, i0]; v111 = data[k1, j1, i1]

    res = np.zeros(3)
    for c in range(3):
        res[c] = ((1-tz)*((1-ty)*((1-tx)*v000[c] + tx*v100[c]) +
                          ty*((1-tx)*v010[c] + tx*v110[c])) +
                   tz*((1-ty)*((1-tx)*v001[c] + tx*v101[c]) +
                       ty*((1-tx)*v011[c] + tx*v111[c])))
    return res


class Grid3D:
    def __init__(self, nx, ny, nz, dx, ox, oy, oz, periodic=False):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.ox = ox
        self.oy = oy
        self.oz = oz
        self.periodic = periodic
        self.vort = np.zeros((nz, ny, nx, 3), dtype=np.float64)
        self.vel  = np.zeros((nz, ny, nx, 3), dtype=np.float64)

    # ---------- 单点访问 ----------
    def set_vort(self, i, j, k, v):
        self.vort[k, j, i, 0] = v.x
        self.vort[k, j, i, 1] = v.y
        self.vort[k, j, i, 2] = v.z

    def get_vort(self, i, j, k) -> Vec3:
        return Vec3(*self.vort[k, j, i])

    def set_vel(self, i, j, k, v):
        self.vel[k, j, i, 0] = v.x
        self.vel[k, j, i, 1] = v.y
        self.vel[k, j, i, 2] = v.z

    def get_vel(self, i, j, k) -> Vec3:
        return Vec3(*self.vel[k, j, i])

    def grid_pos(self, i, j, k) -> Vec3:
        return Vec3(self.ox + (i+0.5)*self.dx,
                    self.oy + (j+0.5)*self.dx,
                    self.oz + (k+0.5)*self.dx)

    # ---------- 单点三线性插值 ----------
    def get_vort_interp(self, p: Vec3) -> Vec3:
        res = trilinear_interp_numba(self.vort, p.x, p.y, p.z,
                                     self.nx, self.ny, self.nz,
                                     self.dx, self.ox, self.oy, self.oz,
                                     self.periodic)
        return Vec3(res[0], res[1], res[2])

    def get_vel_interp(self, p: Vec3) -> Vec3:
        res = trilinear_interp_numba(self.vel, p.x, p.y, p.z,
                                     self.nx, self.ny, self.nz,
                                     self.dx, self.ox, self.oy, self.oz,
                                     self.periodic)
        return Vec3(res[0], res[1], res[2])

    # ---------- 批量三线性插值（向量化） ----------
    def get_vort_interp_batch(self, xs, ys, zs):
        """对数组 (xs,ys,zs) 批量插值涡量，返回 (N,3)"""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        zs = np.asarray(zs, dtype=np.float64)
        N = xs.shape[0]

        fx = (xs - self.ox) / self.dx - 0.5
        fy = (ys - self.oy) / self.dx - 0.5
        fz = (zs - self.oz) / self.dx - 0.5

        if self.periodic:
            fx = fx % self.nx
            fy = fy % self.ny
            fz = fz % self.nz
            i0 = np.floor(fx).astype(np.int32)
            j0 = np.floor(fy).astype(np.int32)
            k0 = np.floor(fz).astype(np.int32)
            i1 = (i0 + 1) % self.nx
            j1 = (j0 + 1) % self.ny
            k1 = (k0 + 1) % self.nz
        else:
            i0 = np.floor(fx).astype(np.int32)
            j0 = np.floor(fy).astype(np.int32)
            k0 = np.floor(fz).astype(np.int32)
            i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
            i0 = np.clip(i0, 0, self.nx-2); i1 = np.clip(i1, 0, self.nx-1)
            j0 = np.clip(j0, 0, self.ny-2); j1 = np.clip(j1, 0, self.ny-1)
            k0 = np.clip(k0, 0, self.nz-2); k1 = np.clip(k1, 0, self.nz-1)

        tx = fx - i0; ty = fy - j0; tz = fz - k0

        v000 = self.vort[k0, j0, i0]   # (N,3)
        v100 = self.vort[k0, j0, i1]
        v010 = self.vort[k0, j1, i0]
        v110 = self.vort[k0, j1, i1]
        v001 = self.vort[k1, j0, i0]
        v101 = self.vort[k1, j0, i1]
        v011 = self.vort[k1, j1, i0]
        v111 = self.vort[k1, j1, i1]

        c000 = (1-tx)*(1-ty)*(1-tz)
        c100 = tx*(1-ty)*(1-tz)
        c010 = (1-tx)*ty*(1-tz)
        c110 = tx*ty*(1-tz)
        c001 = (1-tx)*(1-ty)*tz
        c101 = tx*(1-ty)*tz
        c011 = (1-tx)*ty*tz
        c111 = tx*ty*tz

        return (c000[:,None]*v000 + c100[:,None]*v100 +
                c010[:,None]*v010 + c110[:,None]*v110 +
                c001[:,None]*v001 + c101[:,None]*v101 +
                c011[:,None]*v011 + c111[:,None]*v111)   # (N,3)

    def get_vel_interp_batch(self, xs, ys, zs):
        """批量插值速度，返回 (N,3)"""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        zs = np.asarray(zs, dtype=np.float64)

        fx = (xs - self.ox) / self.dx - 0.5
        fy = (ys - self.oy) / self.dx - 0.5
        fz = (zs - self.oz) / self.dx - 0.5

        if self.periodic:
            fx = fx % self.nx
            fy = fy % self.ny
            fz = fz % self.nz
            i0 = np.floor(fx).astype(np.int32)
            j0 = np.floor(fy).astype(np.int32)
            k0 = np.floor(fz).astype(np.int32)
            i1 = (i0 + 1) % self.nx
            j1 = (j0 + 1) % self.ny
            k1 = (k0 + 1) % self.nz
        else:
            i0 = np.floor(fx).astype(np.int32)
            j0 = np.floor(fy).astype(np.int32)
            k0 = np.floor(fz).astype(np.int32)
            i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
            i0 = np.clip(i0, 0, self.nx-2); i1 = np.clip(i1, 0, self.nx-1)
            j0 = np.clip(j0, 0, self.ny-2); j1 = np.clip(j1, 0, self.ny-1)
            k0 = np.clip(k0, 0, self.nz-2); k1 = np.clip(k1, 0, self.nz-1)

        tx = fx - i0; ty = fy - j0; tz = fz - k0

        v000 = self.vel[k0, j0, i0]
        v100 = self.vel[k0, j0, i1]
        v010 = self.vel[k0, j1, i0]
        v110 = self.vel[k0, j1, i1]
        v001 = self.vel[k1, j0, i0]
        v101 = self.vel[k1, j0, i1]
        v011 = self.vel[k1, j1, i0]
        v111 = self.vel[k1, j1, i1]

        c000 = (1-tx)*(1-ty)*(1-tz)
        c100 = tx*(1-ty)*(1-tz)
        c010 = (1-tx)*ty*(1-tz)
        c110 = tx*ty*(1-tz)
        c001 = (1-tx)*(1-ty)*tz
        c101 = tx*(1-ty)*tz
        c011 = (1-tx)*ty*tz
        c111 = tx*ty*tz

        return (c000[:,None]*v000 + c100[:,None]*v100 +
                c010[:,None]*v010 + c110[:,None]*v110 +
                c001[:,None]*v001 + c101[:,None]*v101 +
                c011[:,None]*v011 + c111[:,None]*v111)

    @property
    def slice_z0(self):
        k0 = self.nz // 2
        return np.sqrt(self.vort[k0,:,:,0]**2 +
                       self.vort[k0,:,:,1]**2 +
                       self.vort[k0,:,:,2]**2)