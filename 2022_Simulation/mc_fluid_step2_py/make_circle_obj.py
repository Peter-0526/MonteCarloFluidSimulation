import os
import numpy as np

def make_circle_obj(center=(0.0, 0.0), radius=0.3, n_segments=64, filename="circle.obj"):
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    vertices = np.stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ], axis=1)

    # 输出到脚本所在目录
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(output_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} 0.0\n")
        for i in range(n_segments):
            f.write(f"l {i + 1} {(i + 1) % n_segments + 1}\n")

    print(f"✅ 已生成: {output_path}")
    return output_path

if __name__ == "__main__":
    path = make_circle_obj()
    print("生成完成。")