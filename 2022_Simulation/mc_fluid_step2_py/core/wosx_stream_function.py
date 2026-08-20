# wosx_stream_function.py
import numpy as np
import wosx

class WoSXStreamFunctionSolver:
    """
    使用 WoSX 求解 2D Poisson 方程：
        ∇²Ψ = ω
    边界条件：
        Ψ = 0  (Dirichlet)
    """
    def __init__(self, obj_file, domain_min, domain_max,
                 n_walks=128, epsilon=1e-3, max_walk_length=1024):
        self.dim = 2
        self.channels = 1
        self.n_walks = n_walks
        self.epsilon = epsilon
        self.max_walk_length = max_walk_length
        self.domain_min = np.array(domain_min, dtype=np.float32)
        self.domain_max = np.array(domain_max, dtype=np.float32)

        # 1. 加载边界网格
        self.positions = wosx.FloatNList(dim=self.dim)
        self.indices = wosx.IntNList(dim=self.dim)
        wosx.Utils.load_boundary_mesh(obj_file, self.positions, self.indices,
                                      dim=self.dim)
        wosx.Utils.flip_orientation(self.indices, dim=self.dim)

        # 2. 创建几何查询
        self.geometric_queries = wosx.Core.GeometricQueries(
            True, self.domain_min, self.domain_max, dim=self.dim)

        # 3. 填充 Dirichlet 边界
        self.dirichlet_handler = wosx.Utils.FcpwDirichletBoundaryHandler(dim=self.dim)
        self.dirichlet_handler.build_acceleration_structure(
            self.positions, self.indices)
        wosx.Utils.populate_geometric_queries_for_dirichlet_boundary(
            self.dirichlet_handler, self.geometric_queries, dim=self.dim)

        # 4. 创建 PDE
        self.pde = wosx.Core.PDE(dim=self.dim, channels=self.channels)
        self.pde.absorption_coeff = 0.0
        self.pde.is_source_constant = False
        self.pde.are_robin_conditions_pure_neumann = True

        # 5. 创建求解器
        self.solver = wosx.Solvers.WalkOnSpheres(
            self.geometric_queries, dim=self.dim, channels=self.channels)

    def solve(self, query_points, omega_grid, grid_min, grid_max):
        """
        query_points: (N,2) float32 查询点坐标
        omega_grid:   (ny,nx) float32 涡量场
        grid_min:     (2,) 涡量网格左下角坐标
        grid_max:     (2,) 涡量网格右上角坐标
        返回: (N,) Ψ 值
        """
        # 设置源项（涡量场）
        omega_buffer = omega_grid.astype(np.float32)
        omega_shape = np.array(omega_grid.shape, dtype=np.int32)
        self.pde.source = wosx.Utils.get_dense_grid_source_callback(
            omega_buffer, omega_shape, grid_min, grid_max,
            dim=self.dim, channels=self.channels)

        # 设置 Dirichlet 边界条件（Ψ=0）
        zeros_buffer = np.zeros_like(omega_buffer)
        self.pde.dirichlet = wosx.Utils.get_dense_grid_dirichlet_callback(
            zeros_buffer, omega_shape, grid_min, grid_max,
            dim=self.dim, channels=self.channels)

        # 创建采样点
        sample_pts = []
        for p in query_points:
            pt = p.astype(np.float32)
            # 创建 SamplePoint（参考 demo 的 create_sample_points）
            # 此处简化：内部点 + 估计解
            sample_pts.append(wosx.Solvers.SamplePoint(
                pt, np.zeros(2, dtype=np.float32),
                wosx.Solvers.SampleType.InDomain,
                wosx.Solvers.EstimationQuantity.Solution,
                1.0, self.epsilon, self.epsilon,
                dim=self.dim, channels=self.channels))
        sample_list = wosx.Solvers.SamplePointList(sample_pts,
                                                   dim=self.dim,
                                                   channels=self.channels)

        # 创建 WalkSettings（参考 demo）
        walk_settings = wosx.Solvers.WalkSettings(
            self.epsilon, 0.0, 0.0, 0.0, np.inf,
            self.max_walk_length, 0, 0, False,
            True, True, False, False, True, False, False)

        # n_walks 列表
        n_walks_list = wosx.IntList([self.n_walks] * len(query_points))

        # 求解
        sample_statistics = wosx.Solvers.create_sample_statistics_list(
            len(query_points), dim=self.dim, channels=self.channels)
        progress_bar = wosx.Utils.ProgressBar(len(query_points))
        report_progress = wosx.Utils.get_report_progress_callback(progress_bar)

        self.solver.solve(self.pde, walk_settings, n_walks_list,
                          sample_list, sample_statistics,
                          False, report_progress)
        progress_bar.finish()

        # 提取解
        psi = np.zeros(len(query_points))
        for i in range(len(query_points)):
            psi[i] = sample_statistics[i].get_estimated_solution()
        return psi