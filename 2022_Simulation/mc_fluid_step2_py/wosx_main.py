# wosx_main.py
import argparse
import sys, os

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.wosx_simulator import Simulator   # <-- 使用 WoSX 优化后的 Simulator
from typing import cast
import core.simulator as base_simulator
from core.geometry import Geometry
from core.types import Vec2
from render.visualizer import save_animation

def main():
    parser = argparse.ArgumentParser(description='Phase 2 (WoSX 加速): 2D with free-slip obstacles')
    parser.add_argument('--nx', type=int, default=64, help='Grid resolution X')
    parser.add_argument('--ny', type=int, default=64, help='Grid resolution Y')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--nmc', type=int, default=128, help='WoS paths per query (WoSX walks)')
    parser.add_argument('--total_time', type=float, default=5.0, help='Total simulation time')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')
    parser.add_argument('--obstacle_radius', type=float, default=0.3, help='Circle obstacle radius')
    parser.add_argument('--obstacle_center_x', type=float, default=0.0)
    parser.add_argument('--obstacle_center_y', type=float, default=0.0)
    # 新增：时间积分方案
    parser.add_argument('--time_scheme', type=str, default='euler',
                        choices=['euler', 'rk2', 'mac'],
                        help='Time integration scheme: euler (1st order), rk2 (midpoint), mac (predictor-corrector)')
    args = parser.parse_args()

    # 创建几何体（包含一个圆形障碍物）
    geo = Geometry(outer_bounds=(-1.0, 1.0, -1.0, 1.0))
    geo.add_circle(Vec2(args.obstacle_center_x, args.obstacle_center_y), args.obstacle_radius)

    # 创建模拟器，传入时间积分方案
    sim = Simulator(nx=args.nx, ny=args.ny, dt=args.dt, geometry=geo,
                    time_scheme=args.time_scheme)
    sim.nmc = args.nmc   # 设置 WoSX 中每条路径的随机游走次数

    # 保存动画
    save_animation(cast(base_simulator.Simulator, sim), total_time=args.total_time,
                   output_dir=args.output_dir,
                   filename='wosx_vorticity_with_obstacle.gif', fps=args.fps)

if __name__ == '__main__':
    main()