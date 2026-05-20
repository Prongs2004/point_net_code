import numpy as np
import matplotlib.pyplot as plt

# 全局字体大小
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 32

# =========================================================
# 读取点云
# =========================================================
txt_path = r"../data/modelnet40_normal_resampled/plant/plant_0001.txt"
points = np.loadtxt(txt_path, delimiter=',').astype(np.float32)
points = points[:, :3]

print("Point Cloud Shape:", points.shape)

# =========================================================
# Sliding Window 参数
# =========================================================
x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
z_min, z_max = np.min(points[:,2]), np.max(points[:,2])

block_size = 0.5
stride = 0.25

cur_x = x_min + 0.3
cur_y = y_min + 0.3

x0, x1 = cur_x, cur_x + block_size
y0, y1 = cur_y, cur_y + block_size

# =========================================================
# 提取窗口内点云
# =========================================================
mask = (
    (points[:,0] >= x0) & (points[:,0] <= x1) &
    (points[:,1] >= y0) & (points[:,1] <= y1)
)
local_points = points[mask]

print("Local Block Shape:", local_points.shape)
if len(local_points) == 0:
    print("当前窗口没有点")
    exit()

# =========================================================
# 定义窗口框顶点和边
# =========================================================
corners = np.array([
    [x0, y0, z_min],[x1, y0, z_min],[x1, y1, z_min],[x0, y1, z_min],
    [x0, y0, z_max],[x1, y0, z_max],[x1, y1, z_max],[x0, y1, z_max],
])
edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
]

# =========================================================
# (1) 第一个独立窗口：原始点云
# =========================================================
plt.figure()
ax1 = plt.subplot(projection='3d')
ax1.scatter(points[:,0], points[:,1], points[:,2], s=2)
ax1.set_title("(a) Original Point Cloud")
ax1.view_init(elev=20, azim=45)

# =========================================================
# (2) 第二个独立窗口：滑动窗口划分
# =========================================================
plt.figure()
ax2 = plt.subplot(projection='3d')
ax2.scatter(points[:,0], points[:,1], points[:,2], s=1, c='lightgray', alpha=0.15)
ax2.scatter(local_points[:,0], local_points[:,1], local_points[:,2], c='red', s=6)
# 画窗口框
for edge in edges:
    p1 = corners[edge[0]]
    p2 = corners[edge[1]]
    ax2.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], c='red', linewidth=2)
ax2.set_title("(b) Sliding Window Partition")
ax2.view_init(elev=20, azim=45)

# =========================================================
# (3) 第三个独立窗口：局部点云块
# =========================================================
plt.figure()
ax3 = plt.subplot(projection='3d')
ax3.scatter(local_points[:,0], local_points[:,1], local_points[:,2], c='red', s=6)
ax3.set_title("(c) Local Point Cloud Block")
ax3.view_init(elev=20, azim=45)

# =========================================================
# (4) 第四个独立窗口：检测结果 + bbox
# =========================================================
plt.figure()
ax4 = plt.subplot(projection='3d')
ax4.scatter(local_points[:,0], local_points[:,1], local_points[:,2], c='blue', s=6)

# 生成bbox
center = np.mean(local_points, axis=0)
size = np.max(local_points, axis=0) - np.min(local_points, axis=0)
cx, cy, cz = center
dx, dy, dz = size

bbox_corners = np.array([
    [cx-dx/2, cy-dy/2, cz-dz/2],
    [cx+dx/2, cy-dy/2, cz-dz/2],
    [cx+dx/2, cy+dy/2, cz-dz/2],
    [cx-dx/2, cy+dy/2, cz-dz/2],
    [cx-dx/2, cy-dy/2, cz+dz/2],
    [cx+dx/2, cy-dy/2, cz+dz/2],
    [cx+dx/2, cy+dy/2, cz+dz/2],
    [cx-dx/2, cy+dy/2, cz+dz/2],
])
bbox_edges = edges
for edge in bbox_edges:
    p1 = bbox_corners[edge[0]]
    p2 = bbox_corners[edge[1]]
    ax4.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], c='red', linewidth=2)

ax4.set_title("(d) Detection Result")
ax4.view_init(elev=20, azim=45)

# 显示所有四个窗口
plt.show()