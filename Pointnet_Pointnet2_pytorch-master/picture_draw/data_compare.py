import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 参数
# =========================================================

TXT_PATH = r"../data/modelnet40_normal_resampled/chair/chair_0001.txt"

VOXEL_SIZE = 0.05
SOR_NEIGHBORS = 10
SOR_STD_RATIO = 2.0

# =========================================================
# 读取点云
# =========================================================

points = np.loadtxt(TXT_PATH, delimiter=',').astype(np.float32)
points = points[:, :3]

print("Original Shape:", points.shape)

# =========================================================
# SOR 去噪
# =========================================================

def statistical_outlier_removal(points):

    if len(points) < SOR_NEIGHBORS + 1:
        return points

    diff = np.expand_dims(points, axis=1) - np.expand_dims(points, axis=0)

    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    knn_dists = np.sort(dist, axis=1)[:, 1:SOR_NEIGHBORS+1]

    mean_dist = np.mean(knn_dists, axis=1)

    thresh = np.mean(mean_dist) + SOR_STD_RATIO * np.std(mean_dist)

    mask = mean_dist < thresh

    return points[mask]

# =========================================================
# Voxel 下采样
# =========================================================

def voxel_downsample(points):

    coords = np.floor(points / VOXEL_SIZE).astype(np.int32)

    _, unique_idx = np.unique(coords, axis=0, return_index=True)

    return points[unique_idx]

# =========================================================
# 处理
# =========================================================

sor_points = statistical_outlier_removal(points)
voxel_points = voxel_downsample(sor_points)

print("SOR Shape:", sor_points.shape)
print("Voxel Shape:", voxel_points.shape)

# =========================================================
# 🔥 窗口 1：原始点云
# =========================================================
plt.figure()
ax1 = plt.subplot(projection='3d')
ax1.scatter(points[:,0], points[:,1], points[:,2], s=1, c='blue')
ax1.set_title("Original Point Cloud", fontsize=32)  # 标题32
ax1.view_init(elev=20, azim=45)

# =========================================================
# 🔥 窗口 2：SOR去噪后
# =========================================================
plt.figure()
ax2 = plt.subplot(projection='3d')
ax2.scatter(sor_points[:,0], sor_points[:,1], sor_points[:,2], s=1, c='green')
ax2.set_title("After SOR Filtering", fontsize=32)  # 标题32
ax2.view_init(elev=20, azim=45)

# =========================================================
# 🔥 窗口 3：Voxel下采样后
# =========================================================
plt.figure()
ax3 = plt.subplot(projection='3d')
ax3.scatter(voxel_points[:,0], voxel_points[:,1], voxel_points[:,2], s=2, c='red')
ax3.set_title("After Voxel Downsampling", fontsize=32)  # 标题32
ax3.view_init(elev=20, azim=45)

# 显示所有窗口
plt.show()