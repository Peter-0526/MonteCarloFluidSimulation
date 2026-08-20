# core/wosx_simulator.py
"""
WoSX 加速模拟器

支持时间积分方案：
- euler : 一阶半拉格朗日（原版）
- rk2   : 二阶中点法 (RK2)
- mac   : 预测-校正法（类似 MacCormack）
"""
import os
import numpy as np
import wosx
from .grid import Grid
from .geometry import Geometry
from .types import Vec2, RNG
from typing import Optional


def generate_circle_obj(center, radius, filename, segments=64):
    """生成圆形的 .obj 边界文件（2D 线段）"""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    vertices = np.stack([
        center.x + radius * np.cos(angles),
        center.y + radius * np.sin(angles)
    ], axis=1)
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.0\n")
        for i in range(segments):
            f.write(f"l {i+1} {(i+1) % segments + 1}\n")
    print(f"生成边界文件: {filename}")


def generate_polygon_obj(vertices, filename):
    """将多边形顶点列表写为闭合线段 .obj"""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v.x:.6f} {v.y:.6f} 0.0\n")
        n = len(vertices)
        for i in range(n):
            f.write(f"l {i+1} {(i+1) % n + 1}\n")
    print(f"生成边界文件: {filename}")


class Simulator:
    """蒙特卡洛流体模拟器，使用 WoSX 求解流函数（Poisson）以重构速度场"""

    def __init__(self, nx, ny, dt, geometry: Optional[Geometry] = None,
                 time_scheme: str = 'euler'):
        """
        Parameters:
        -----------
        time_scheme : str
            时间推进方案：'euler'（一阶）, 'rk2'（二阶中点）, 'mac'（预测-校正）
        """
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = 2.0 / nx
        ox, oy = -1.0, -1.0
        self.grid = [Grid(nx, ny, self.dx, ox, oy) for _ in range(2)]
        self.cur = 0
        self.nmc = 128
        self.paused = False
        self.rng = RNG()
        self.geometry = geometry if geometry is not None else Geometry()

        # 时间积分方案
        scheme = time_scheme.lower()
        if scheme not in ('euler', 'rk2', 'mac'):
            print(f"警告: 未知的时间积分方案 '{time_scheme}'，使用 euler")
            scheme = 'euler'
        self.time_scheme = scheme

        if len(self.geometry.obstacles) > 0:
            self._init_wosx()
            self.use_wosx = True
        else:
            self.use_wosx = False
            from .biot_savart import BiotSavart
            self.biot_savart = BiotSavart

        self._init_vorticity()

    def _init_wosx(self):
        obj_path = os.path.join(os.path.dirname(__file__), "boundary.obj")
        if not os.path.exists(obj_path):
            # 支持圆形和正方形
            for obs in self.geometry.obstacles:
                if obs[0] == 'circle':
                    generate_circle_obj(obs[1], obs[2], obj_path)
                    break
                elif obs[0] == 'polygon':
                    generate_polygon_obj(obs[1], obj_path)
                    break
            else:
                raise ValueError("未找到障碍物")

        # 2. 加载边界并创建几何查询
        self.positions = wosx.FloatNList(dim=2)
        self.indices = wosx.IntNList(dim=2)
        wosx.Utils.load_boundary_mesh(obj_path, self.positions, self.indices, dim=2)
        wosx.Utils.flip_orientation(self.indices, dim=2)

        domain_min = np.array([-1.0, -1.0], dtype=np.float32)
        domain_max = np.array([1.0, 1.0], dtype=np.float32)
        self.geometric_queries = wosx.Core.GeometricQueries(True, domain_min, domain_max, dim=2)

        # 3. 填充 Dirichlet 边界（Ψ=0）
        self.dirichlet_handler = wosx.Utils.FcpwDirichletBoundaryHandler(dim=2)
        self.dirichlet_handler.build_acceleration_structure(self.positions, self.indices)
        wosx.Utils.populate_geometric_queries_for_dirichlet_boundary(
            self.dirichlet_handler, self.geometric_queries, dim=2)

        # 4. 创建 PDE
        self.pde = wosx.Core.PDE(dim=2, channels=1)
        self.pde.absorption_coeff = 0.0
        self.pde.is_source_constant = False
        self.pde.are_robin_conditions_pure_neumann = True

        # 5. 创建 WoS 求解器
        self.solver = wosx.Solvers.WalkOnSpheres(self.geometric_queries, dim=2, channels=1)

    def _solve_psi(self, query_points, omega_grid):
        """使用 WoSX 批量求解流函数 Ψ"""
        n = len(query_points)

        omega_buffer = np.ascontiguousarray(omega_grid, dtype=np.float32).ravel()
        omega_shape = np.array([omega_grid.shape[0], omega_grid.shape[1]], dtype=np.int32)
        grid_min = np.array([self.grid[0].ox, self.grid[0].oy], dtype=np.float32)
        grid_max = np.array([self.grid[0].ox + self.nx * self.dx,
                             self.grid[0].oy + self.ny * self.dx], dtype=np.float32)

        self.pde.source = wosx.Utils.get_dense_grid_source_callback(
            omega_buffer, omega_shape, grid_min, grid_max,
            enable_interpolation=True, dim=2, channels=1)

        zero_buffer = np.zeros_like(omega_buffer)
        self.pde.dirichlet = wosx.Utils.get_dense_grid_dirichlet_callback(
            zero_buffer, omega_shape, grid_min, grid_max,
            enable_interpolation=True, dim=2, channels=1)

        solve_locations = np.array(query_points, dtype=np.float32)
        dist_to_absorbing = wosx.FloatList()
        dist_to_reflecting = wosx.FloatList()
        wosx.Utils.compute_dist_to_boundary(
            self.geometric_queries,
            solve_locations,
            dist_to_absorbing,
            dist_to_reflecting,
            dim=2)

        sample_pts = []
        for i, p in enumerate(query_points):
            pt = np.asarray(p, dtype=np.float32)
            sample_pts.append(wosx.Solvers.SamplePoint(
                pt, np.zeros(2, dtype=np.float32),
                wosx.Solvers.SampleType.InDomain,
                wosx.Solvers.EstimationQuantity.Solution,
                1.0,
                dist_to_absorbing[i],
                dist_to_reflecting[i],
                dim=2, channels=1))
        sample_list = wosx.Solvers.SamplePointList(sample_pts, dim=2, channels=1)

        walk_settings = wosx.Solvers.WalkSettings(
            epsilon_shell_for_absorbing_boundary=1e-3,
            epsilon_shell_for_reflecting_boundary=0.0,
            silhouette_precision=0.0,
            russian_roulette_threshold=0.0,
            splitting_threshold=np.inf,
            max_walk_length=1024,
            steps_before_applying_tikhonov=0,
            steps_before_using_maximal_spheres=0,
            solve_double_sided=False,
            use_gradient_control_variates=True,
            use_gradient_antithetic_variates=True,
            use_cosine_sampling_for_derivatives=False,
            ignore_absorbing_boundary_contribution=False,
            ignore_reflecting_boundary_contribution=True,
            ignore_source_contribution=False,
            print_logs=False)

        n_walks_list = wosx.IntList([self.nmc] * n)
        stats = wosx.Solvers.create_sample_statistics_list(n, dim=2, channels=1)

        progress_bar = wosx.Utils.ProgressBar(n)
        report_progress = wosx.Utils.get_report_progress_callback(progress_bar)
        self.solver.solve(self.pde, walk_settings, n_walks_list,
                          sample_list, stats, False, report_progress)

        psi = np.zeros(n)
        for i in range(n):
            psi[i] = stats[i].get_estimated_solution()
        return psi

    def _init_vorticity(self):
        """初始化两个高斯涡旋（避开障碍物）"""
        sigma = 0.2
        centers = [Vec2(-0.5, 0.0), Vec2(0.5, 0.0)]
        for j in range(self.ny):
            for i in range(self.nx):
                p = self.grid[0].grid_pos(i, j)
                if self.geometry.inside_domain(p):
                    w = 0.0
                    for c in centers:
                        d = p - c
                        w += np.exp(-d.norm2() / (2 * sigma * sigma))
                else:
                    w = 0.0
                self.grid[0].set_vort(i, j, w)
                self.grid[1].set_vort(i, j, w)

    # ---------- 双线性插值辅助函数 ----------
    def _bilinear_interp(self, field, px, py):
        """在 (px, py) 处双线性插值 field (2D array)"""
        fx = (px - self.grid[0].ox) / self.dx - 0.5
        fy = (py - self.grid[0].oy) / self.dx - 0.5
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        tx = fx - i0
        ty = fy - j0
        i0 = max(0, min(i0, self.nx - 2))
        j0 = max(0, min(j0, self.ny - 2))
        i1, j1 = i0 + 1, j0 + 1
        v00 = field[j0, i0]
        v10 = field[j0, i1]
        v01 = field[j1, i0]
        v11 = field[j1, i1]
        return ((1 - ty) * ((1 - tx) * v00 + tx * v10) +
                ty * ((1 - tx) * v01 + tx * v11))

    def _interp_velocity(self, vx, vy, px, py):
        """插值速度，返回 Vec2"""
        vx_i = self._bilinear_interp(vx, px, py)
        vy_i = self._bilinear_interp(vy, px, py)
        return Vec2(vx_i, vy_i)

    def _compute_velocity_from_psi(self, psi_grid):
        """从流函数计算速度场 (vx, vy)，中心差分"""
        dpsi_dx = np.zeros_like(psi_grid)
        dpsi_dy = np.zeros_like(psi_grid)
        if self.nx > 2:
            dpsi_dx[:, 1:-1] = (psi_grid[:, 2:] - psi_grid[:, :-2]) / (2 * self.dx)
        if self.ny > 2:
            dpsi_dy[1:-1, :] = (psi_grid[2:, :] - psi_grid[:-2, :]) / (2 * self.dx)
        vx = -dpsi_dy
        vy = dpsi_dx
        return vx, vy

    def _get_domain_points(self):
        """返回域内点的坐标列表和网格索引列表"""
        query_points = []
        point_indices = []
        for j in range(self.ny):
            for i in range(self.nx):
                x = self.grid[self.cur].grid_pos(i, j)
                if self.geometry.inside_domain(x):
                    query_points.append([x.x, x.y])
                    point_indices.append((i, j))
        return query_points, point_indices

    def _advect_back(self, x, vx, vy, time_scheme, vx2=None, vy2=None):
        """
        根据速度场计算回溯点 x_b
        time_scheme: 'euler', 'rk2', 'mac'
        vx, vy: 速度场矩阵（或 None 如果使用插值）
        vx2, vy2: 第二速度场（用于 mac 平均）
        """
        dt = self.dt
        if time_scheme == 'euler':
            v = self._interp_velocity(vx, vy, x.x, x.y)
            xb = x - v * dt
        elif time_scheme == 'rk2':
            v1 = self._interp_velocity(vx, vy, x.x, x.y)
            x_mid = x - v1 * (0.5 * dt)
            # 确保中点仍在域内（否则微调）
            if not self.geometry.inside_domain(x_mid):
                x_mid = self.geometry.closest_point(x_mid)
                inward = x_mid - x
                if inward.norm() > 1e-8:
                    x_mid = x_mid + inward * 1e-4
            v2 = self._interp_velocity(vx, vy, x_mid.x, x_mid.y)
            xb = x - v2 * dt
        elif time_scheme == 'mac':
            # 平均速度: 第一速度场和第二速度场（如果有）
            v1 = self._interp_velocity(vx, vy, x.x, x.y)
            if vx2 is not None:
                v2 = self._interp_velocity(vx2, vy2, x.x, x.y)
                v_avg = (v1 + v2) * 0.5
            else:
                v_avg = v1
            xb = x - v_avg * dt
        else:
            raise ValueError(f"Unknown time_scheme: {time_scheme}")

        # 确保回溯点在域内
        if not self.geometry.inside_domain(xb):
            xb = self.geometry.closest_point(xb)
            inward = xb - x
            if inward.norm() > 1e-8:
                xb = xb + inward * 1e-4
        return xb

    def step(self):
        prev = self.grid[self.cur]
        nxt = self.grid[1 - self.cur]

        if not self.use_wosx:
            # 无边界时使用 Biot-Savart（简单处理：仅支持 euler / rk2）
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i, j)
                    if not self.geometry.inside_domain(x):
                        nxt.set_vort(i, j, 0.0)
                        continue
                    # 先获得速度（BiotSavart 逐点）
                    v = self.biot_savart.estimate_velocity(prev, x, self.nmc, self.rng)
                    if self.time_scheme == 'rk2':
                        x_mid = x - v * (0.5 * self.dt)
                        if not self.geometry.inside_domain(x_mid):
                            x_mid = self.geometry.closest_point(x_mid)
                            inward = x_mid - x
                            if inward.norm() > 1e-8:
                                x_mid = x_mid + inward * 1e-4
                        v2 = self.biot_savart.estimate_velocity(prev, x_mid, self.nmc, self.rng)
                        xb = x - v2 * self.dt
                    else:  # euler (mac 不支持 BiotSavart，回退到 euler)
                        xb = x - v * self.dt
                    if not self.geometry.inside_domain(xb):
                        xb = self.geometry.closest_point(xb)
                        inward = xb - x
                        if inward.norm() > 1e-8:
                            xb = xb + inward * 1e-4
                    w = prev.get_vort(xb)
                    nxt.set_vort(i, j, w)
            self.cur = 1 - self.cur
            return

        # ---------- WoSX 分支 ----------
        # 1. 域内点
        query_points, point_indices = self._get_domain_points()

        # 2. 求解当前时刻的流函数
        psi_vals = self._solve_psi(query_points, prev.vort)
        psi_grid = np.zeros((self.ny, self.nx))
        for idx, (i, j) in enumerate(point_indices):
            psi_grid[j, i] = psi_vals[idx]

        vx, vy = self._compute_velocity_from_psi(psi_grid)

        # 根据 time_scheme 更新涡量
        if self.time_scheme == 'euler':
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i, j)
                    if not self.geometry.inside_domain(x):
                        nxt.set_vort(i, j, 0.0)
                        continue
                    xb = self._advect_back(x, vx, vy, 'euler')
                    w = prev.get_vort(xb)
                    nxt.set_vort(i, j, w)

        elif self.time_scheme == 'rk2':
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i, j)
                    if not self.geometry.inside_domain(x):
                        nxt.set_vort(i, j, 0.0)
                        continue
                    xb = self._advect_back(x, vx, vy, 'rk2')
                    w = prev.get_vort(xb)
                    nxt.set_vort(i, j, w)

        elif self.time_scheme == 'mac':
            # 预测步：用当前速度回溯得到预测涡量 w_pred
            w_pred = np.zeros((self.ny, self.nx))
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i, j)
                    if not self.geometry.inside_domain(x):
                        w_pred[j, i] = 0.0
                        continue
                    xb = self._advect_back(x, vx, vy, 'euler')  # 用一阶预测
                    w_pred[j, i] = prev.get_vort(xb)

            # 使用预测涡量求解新速度场
            psi_pred_vals = self._solve_psi(query_points, w_pred)
            psi_pred_grid = np.zeros((self.ny, self.nx))
            for idx, (i, j) in enumerate(point_indices):
                psi_pred_grid[j, i] = psi_pred_vals[idx]
            vx_pred, vy_pred = self._compute_velocity_from_psi(psi_pred_grid)

            # 校正步：用平均速度回溯得到最终值
            for j in range(self.ny):
                for i in range(self.nx):
                    x = prev.grid_pos(i, j)
                    if not self.geometry.inside_domain(x):
                        nxt.set_vort(i, j, 0.0)
                        continue
                    xb = self._advect_back(x, vx, vy, 'mac', vx_pred, vy_pred)
                    w = prev.get_vort(xb)
                    nxt.set_vort(i, j, w)

        self.cur = 1 - self.cur

    @property
    def vorticity(self):
        return self.grid[self.cur].vort

    def reset(self):
        self.cur = 0
        self._init_vorticity()