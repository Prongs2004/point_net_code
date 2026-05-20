import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 读取点云
# =========================================================

txt_path = r"../data/modelnet40_normal_resampled/plant/plant_0002.txt"

points = np.loadtxt(txt_path, delimiter=',').astype(np.float32)

points = points[:, :3]
print("Point Cloud Shape:", points.shape)

# =========================================================
# 自动生成bbox
# =========================================================

xyz_min = np.min(points, axis=0)
xyz_max = np.max(points, axis=0)

center = (xyz_min + xyz_max) / 2
size = xyz_max - xyz_min

cx, cy, cz = center
dx, dy, dz = size

# =========================================================
# 创建多个重复bbox（模拟NMS前）
# =========================================================

boxes_before = []

offsets = [
    [0.00, 0.00, 0.00],
    [0.05, 0.02, 0.00],
    [-0.04, 0.03, 0.01],
    [0.03, -0.05, -0.02],
]

for off in offsets:
    ox, oy, oz = off
    box = [cx + ox, cy + oy, cz + oz, dx, dy, dz]
    boxes_before.append(box)

# NMS后仅保留一个框
boxes_after = [[cx, cy, cz, dx, dy, dz]]

# =========================================================
# bbox绘制函数
# =========================================================

def draw_bbox(ax, box, color='red'):
    cx, cy, cz, dx, dy, dz = box

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

    for edge in edges:
        p1 = corners[edge[0]]
        p2 = corners[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, linewidth=2)

# =========================================================
# 🔥 第一个独立窗口：NMS 前
# =========================================================
plt.figure()
ax1 = plt.subplot(projection='3d')
ax1.scatter(points[:,0], points[:,1], points[:,2], s=2, c='blue', alpha=0.4)
for box in boxes_before:
    draw_bbox(ax1, box)
ax1.set_title("Before NMS", fontsize=32)  # 标题32号
ax1.view_init(elev=20, azim=45)

# =========================================================
# 🔥 第二个独立窗口：NMS 后
# =========================================================
plt.figure()
ax2 = plt.subplot(projection='3d')
ax2.scatter(points[:,0], points[:,1], points[:,2], s=2, c='blue', alpha=0.4)
for box in boxes_after:
    draw_bbox(ax2, box)
ax2.set_title("After NMS", fontsize=32)  # 标题32号
ax2.view_init(elev=20, azim=45)

# 显示两个独立窗口
plt.show()