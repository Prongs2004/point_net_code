import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 读取点云
# =========================================================

txt_path = r"../data/modelnet40_normal_resampled/plant/plant_0001.txt"

points = np.loadtxt(txt_path, delimiter=',').astype(np.float32)

points = points[:, :3]

print("Point Cloud Shape:", points.shape)

# =========================================================
# 自动生成 bbox
# （模拟检测结果）
# =========================================================

xyz_min = np.min(points, axis=0)
xyz_max = np.max(points, axis=0)

# bbox中心与尺寸
center = (xyz_min + xyz_max) / 2
size = xyz_max - xyz_min

cx, cy, cz = center
dx, dy, dz = size

# =========================================================
# (a) 原始点云 —— 独立窗口 1
# =========================================================
plt.figure()
ax1 = plt.subplot(projection='3d')

ax1.scatter(
    points[:,0],
    points[:,1],
    points[:,2],
    s=2
)
ax1.set_title("(a) Original Point Cloud", fontsize=32)
ax1.view_init(elev=20, azim=45)

# =========================================================
# (b) bbox检测结果 —— 独立窗口 2
# =========================================================
plt.figure()
ax2 = plt.subplot(projection='3d')

ax2.scatter(
    points[:,0],
    points[:,1],
    points[:,2],
    s=2,
    c='lightgray',
    alpha=0.4
)

# =========================================================
# bbox顶点
# =========================================================
corners = np.array([
    [cx-dx/2, cy-dy/2, cz-dz/2],
    [cx+dx/2, cy-dy/2, cz-dz/2],
    [cx+dx/2, cy+dy/2, cz-dz/2],
    [cx-dx/2, cy+dy/2, cz-dz/2],

    [cx-dx/2, cy-dy/2, cz+dz/2],
    [cx+dx/2, cy-dy/2, cz+dz/2],
    [cx+dx/2, cy+dy/2, cz+dz/2],
    [cx-dx/2, cy+dy/2, cz+dz/2],
])

edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
]

# =========================================================
# 绘制bbox
# =========================================================
for edge in edges:
    p1 = corners[edge[0]]
    p2 = corners[edge[1]]

    ax2.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        c='red',
        linewidth=2
    )

# =========================================================
# bbox中心点
# =========================================================
ax2.scatter(
    cx, cy, cz,
    s=80,
    c='blue'
)

# =========================================================
# 添加类别标签
# =========================================================
ax2.text(
    cx,
    cy,
    cz,
    'plant',    # 把 chair 改成 plant 更准确
    fontsize=12
)

ax2.set_title("(b) 3D Bounding Box Detection", fontsize=32)
ax2.view_init(elev=20, azim=45)

# 显示两个独立窗口
plt.show()