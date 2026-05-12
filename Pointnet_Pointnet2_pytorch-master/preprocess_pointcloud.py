import torch
import numpy as np
import os
from tqdm import tqdm

# ====================== 【你的论文创新参数，完全保留】 ======================
NUM_POINTS = 1024
VOXEL_SIZE = 0.02        # 精细下采样，保留细节
SOR_NEIGHBORS = 8        # SOR去噪邻域数
SOR_STD_RATIO = 1.5      # 温和去噪，不误删结构
# ===========================================================================

# 设备配置：自动使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_gpu_preprocess')

def load_pointcloud(file_path):
    """加载点云"""
    try:
        points = np.loadtxt(file_path, delimiter=',')[:, :3]
        return points.astype(np.float32)
    except:
        return None

def normalize_pointcloud_gpu(points):
    """【GPU】单位球归一化（PointNet官方标配）"""
    points = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    centroid = torch.mean(points, dim=0)
    points = points - centroid
    max_dist = torch.max(torch.norm(points, dim=1))
    points = points / max_dist
    return points

def sor_denoise_gpu(points, k=8, std_ratio=1.5):
    """【GPU加速】SOR去噪（你的核心创新，逻辑完全一致）"""
    N = points.shape[0]
    if N <= k:
        return points

    # GPU 快速 KNN 计算（比numpy快100倍）
    dist = torch.cdist(points, points)  # GPU 矩阵距离计算
    knn_dists = dist.topk(k+1, dim=1, largest=False, sorted=True)[0][:, 1:]
    mean_dist = torch.mean(knn_dists, dim=1)

    # 计算阈值并过滤
    thresh = torch.mean(mean_dist) + std_ratio * torch.std(mean_dist)
    mask = mean_dist < thresh
    return points[mask]

def voxel_downsample_gpu(points):
    """【GPU加速】体素下采样（你的核心创新）"""
    coords = torch.floor(points / VOXEL_SIZE).long()
    # 去重：保留每个体素的第一个点
    unique_coords, indices = torch.unique(coords, dim=0, return_inverse=True)
    idx = torch.zeros(unique_coords.shape[0], dtype=torch.long, device=DEVICE)
    idx[indices] = torch.arange(points.shape[0], device=DEVICE)
    return points[idx]

def fps_gpu(points, n_samples):
    """【GPU加速】最远点采样（保留几何结构）"""
    N = points.shape[0]
    if N >= n_samples:
        idx = torch.zeros(n_samples, dtype=torch.long, device=DEVICE)
        distance = torch.ones(N, device=DEVICE) * 1e10
        farthest = torch.randint(0, N, (1,), device=DEVICE)

        for i in range(n_samples):
            idx[i] = farthest
            dist = torch.sum((points - points[farthest]) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance)

        return points[idx]
    else:
        choice = torch.randint(0, N, (n_samples,), device=DEVICE)
        return points[choice]

def preprocess_dataset_gpu():
    """GPU 主预处理流程"""
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    total_saved = 0
    total_skipped = 0

    # 遍历所有类别
    for cls in tqdm(os.listdir(DATA_DIR), desc="总进度"):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        out_cls_path = os.path.join(OUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)

        # 遍历每个点云文件
        for file_name in os.listdir(cls_path):
            if not file_name.endswith('.txt'):
                continue

            file_path = os.path.join(cls_path, file_name)
            points = load_pointcloud(file_path)
            if points is None:
                total_skipped += 1
                continue

            # ====================== 【你的核心创新流程，完整保留】 ======================
            points = normalize_pointcloud_gpu(points)   # 归一化
            points = sor_denoise_gpu(points, SOR_NEIGHBORS, SOR_STD_RATIO)  # GPU去噪
            points = voxel_downsample_gpu(points)         # GPU下采样
            points = fps_gpu(points, NUM_POINTS)          # GPU采样

            # 转回numpy保存
            points_np = points.cpu().numpy()
            save_path = os.path.join(out_cls_path, file_name.replace('.txt', '.npy'))
            np.save(save_path, points_np)

            total_saved += 1

    print(f"\n🎉 GPU预处理完成！")
    print(f"✅ 成功处理: {total_saved} 个点云")
    print(f"✅ 跳过文件: {total_skipped}")
    print(f"✅ 输出目录: {OUT_DIR}")

if __name__ == '__main__':
    preprocess_dataset_gpu()