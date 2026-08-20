#!/usr/bin/env python3
"""
WoSX 库功能验证脚本
运行方式：python test_wosx.py
"""

import sys
import numpy as np

print("=" * 70)
print("步骤 1：导入 wosx")
print("=" * 70)

try:
    import wosx
    print("✅ wosx 导入成功")
    print("wosx 模块位置:", wosx.__file__)
except ImportError as e:
    print("❌ wosx 导入失败:", e)
    sys.exit(1)


print("\n" + "=" * 70)
print("步骤 2：检查顶层类型")
print("=" * 70)

top_level_names = [
    "BoolList", "IntList", "UintList", "FloatList",
    "Float2List", "Float3List", "Int2List", "Int3List",
    "FloatNList", "IntNList", "convert_list_to_numpy_array",
    "Core", "Solvers", "Samplers", "Utils",
]

for name in top_level_names:
    if hasattr(wosx, name):
        obj = getattr(wosx, name)
        print(f"✅ {name:30s} -> {type(obj).__name__}")
    else:
        print(f"❌ {name:30s} -> 缺失")

# GPU 支持检查
has_gpu = hasattr(wosx, "GPUTaskHandle")
print(f"\nGPU 支持: {'✅ 可用' if has_gpu else '❌ 不可用（可能是 CPU-only 安装）'}")


print("\n" + "=" * 70)
print("步骤 3：测试 FloatNList / IntNList 分发函数")
print("=" * 70)

try:
    flist2 = wosx.FloatNList([1.0, 2.0, 3.0], dim=2)   # 2D 点列表
    print("✅ FloatNList(dim=2) 创建成功, 长度:", len(flist2))
except Exception as e:
    print("❌ FloatNList(dim=2) 失败:", e)

try:
    ilist2 = wosx.IntNList([0, 1, 2, 3], dim=2)
    print("✅ IntNList(dim=2) 创建成功, 长度:", len(ilist2))
except Exception as e:
    print("❌ IntNList(dim=2) 失败:", e)


print("\n" + "=" * 70)
print("步骤 4：检查 Core 子模块")
print("=" * 70)

core_members = [n for n in dir(wosx.Core) if not n.startswith('_')]
print("Core 公共成员数量:", len(core_members))
print("Core 成员:", core_members[:30], "...")

# 尝试创建 Core.GeometricQueries (dim=2)
try:
    sq = wosx.Core.GeometricQueries(dim=2)
    print("✅ Core.GeometricQueries(dim=2) 创建成功")
except Exception as e:
    print("❌ Core.GeometricQueries(dim=2) 失败:", e)

# 尝试创建 Core.PDE (dim=2, channels=1)
try:
    pde = wosx.Core.PDE(dim=2, channels=1)
    print("✅ Core.PDE(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Core.PDE(dim=2, channels=1) 失败:", e)


print("\n" + "=" * 70)
print("步骤 5：检查 Solvers 子模块")
print("=" * 70)

# 尝试创建 WalkOnSpheres 求解器
try:
    wos_solver = wosx.Solvers.WalkOnSpheres(dim=2, channels=1)
    print("✅ Solvers.WalkOnSpheres(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Solvers.WalkOnSpheres 失败:", e)

# 尝试创建 SampleStatistics
try:
    stats = wosx.Solvers.SampleStatistics(dim=2, channels=1)
    print("✅ Solvers.SampleStatistics(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Solvers.SampleStatistics 失败:", e)

# 尝试创建 SamplePointList
try:
    points = wosx.Solvers.SamplePointList(dim=2, channels=1)
    print("✅ Solvers.SamplePointList(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Solvers.SamplePointList 失败:", e)


print("\n" + "=" * 70)
print("步骤 6：检查 Samplers 子模块")
print("=" * 70)

try:
    boundary_sampler = wosx.Samplers.BoundarySampler(dim=2, channels=1)
    print("✅ Samplers.BoundarySampler(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Samplers.BoundarySampler 失败:", e)

try:
    domain_sampler = wosx.Samplers.DomainSampler(dim=2, channels=1)
    print("✅ Samplers.DomainSampler(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Samplers.DomainSampler 失败:", e)


print("\n" + "=" * 70)
print("步骤 7：检查 Utils 子模块")
print("=" * 70)

# 尝试创建边界处理器（Dirichlet）
try:
    handler = wosx.Utils.FcpwDirichletBoundaryHandler(dim=2)
    print("✅ Utils.FcpwDirichletBoundaryHandler(dim=2) 创建成功")
except Exception as e:
    print("❌ Utils.FcpwDirichletBoundaryHandler(dim=2) 失败:", e)

# 检查 DenseGrid
try:
    grid = wosx.Utils.FloatDenseGrid(dim=2, channels=1)
    print("✅ Utils.FloatDenseGrid(dim=2, channels=1) 创建成功")
except Exception as e:
    print("❌ Utils.FloatDenseGrid 失败:", e)


print("\n" + "=" * 70)
print("步骤 8：查看关键类的帮助信息")
print("=" * 70)

for name, obj in [
    ("Core.PDE", wosx.Core.PDE),
    ("Solvers.WalkOnSpheres", wosx.Solvers.WalkOnSpheres),
    ("Samplers.create_uniform_line_segment_boundary_sampler",
     wosx.Samplers.create_uniform_line_segment_boundary_sampler),
]:
    try:
        help_text = obj.__doc__ or "（无文档）"
        print(f"\n--- {name} ---")
        print(help_text[:1000])
    except Exception as e:
        print(f"无法获取 {name} 的帮助:", e)


print("\n" + "=" * 70)
print("测试完成，WoSX 基础功能验证结束")
print("=" * 70)