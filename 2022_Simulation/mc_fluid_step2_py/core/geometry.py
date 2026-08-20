import numpy as np
from .types import Vec2

class Geometry:
    """ 管理外边界与内部固体障碍物的距离和最近点查询 """
    def __init__(self, outer_bounds=(-1.0, 1.0, -1.0, 1.0)):
        self.outer = outer_bounds        # (xmin, xmax, ymin, ymax)
        self.obstacles = []              # list of (center, radius) or (p0, p1) ...
    
    def add_circle(self, center: Vec2, radius: float):
        self.obstacles.append(('circle', center, radius))
    
    def distance(self, p: Vec2) -> float:
        """ 到最近的边界（外边界或障碍物）的距离 """
        d_out = min(p.x - self.outer[0], self.outer[1] - p.x,
                    p.y - self.outer[2], self.outer[3] - p.y)
        d_obs = float('inf')
        for obst in self.obstacles:
            if obst[0] == 'circle':
                c, r = obst[1], obst[2]
                d_obs = min(d_obs, max(0.0, (p - c).norm() - r))
        return min(d_out, d_obs)
    
    def closest_point(self, p: Vec2) -> Vec2:
        """ 返回最近的边界点（用于 Dirichlet 条件，Ψ=0） """
        # 先考虑外边界（轴对齐矩形）
        # x 方向
        if p.x < self.outer[0]:
            x = self.outer[0]
        elif p.x > self.outer[1]:
            x = self.outer[1]
        else:
            x = p.x
        # y 方向
        if p.y < self.outer[2]:
            y = self.outer[2]
        elif p.y > self.outer[3]:
            y = self.outer[3]
        else:
            y = p.y
        best = Vec2(x, y)
        d_best = (best - p).norm()
        # 考虑障碍物
        for obst in self.obstacles:
            if obst[0] == 'circle':
                c, r = obst[1], obst[2]
                dir_vec = (p - c)
                dist = dir_vec.norm()
                if dist > 1e-12:
                    cp = c + dir_vec * (r / dist)
                else:
                    cp = c + Vec2(r, 0)   # fallback
                d_cp = (cp - p).norm()
                if d_cp < d_best:
                    best = cp
                    d_best = d_cp
        return best
    
    def inside_domain(self, p: Vec2) -> bool:
        """ 检查点是否在域内（外边界内且不在障碍物内） """
        if not (self.outer[0] <= p.x <= self.outer[1] and 
                self.outer[2] <= p.y <= self.outer[3]):
            return False
        for obst in self.obstacles:
            if obst[0] == 'circle':
                c, r = obst[1], obst[2]
                if (p - c).norm() < r - 1e-8:
                    return False
        return True

    # core/geometry.py（添加方法）
    def add_rectangle(self, center_x, center_y, width, height):
        """添加一个轴对齐矩形障碍物"""
        half_w = width / 2.0
        half_h = height / 2.0
        corners = [
            Vec2(center_x - half_w, center_y - half_h),
            Vec2(center_x + half_w, center_y - half_h),
            Vec2(center_x + half_w, center_y + half_h),
            Vec2(center_x - half_w, center_y + half_h),
        ]
        # 将矩形保存为多边形（障碍物列表）
        self.obstacles.append(('polygon', corners))