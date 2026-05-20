import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 加载点云
# =========================================================

chair = np.loadtxt(
    "../data/modelnet40_normal_resampled/chair/chair_0001.txt",
    delimiter=','
)[:, :3]

table = np.loadtxt(
    "../data/modelnet40_normal_resampled/table/table_0001.txt",
    delimiter=','
)[:, :3]

sofa = np.loadtxt(
    "../data/modelnet40_normal_resampled/sofa/sofa_0001.txt",
    delimiter=','
)[:, :3]

# =========================================================
# 平移位置（关键）
# =========================================================

chair[:, 0] -= 1.5

table[:, 0] += 0

sofa[:, 0] += 1.5

# =========================================================
# 拼接场景
# =========================================================

scene = np.concatenate([
    chair,
    table,
    sofa
], axis=0)

print("Scene Shape:", scene.shape)

# =========================================================
# 可视化
# =========================================================

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    scene[:,0],
    scene[:,1],
    scene[:,2],
    s=2,
    c='blue'
)

# =========================================================
# 标题
# =========================================================

ax.set_title(
    "Pseudo 3D Indoor Scene",
    fontsize=14
)

plt.tight_layout()

plt.show()