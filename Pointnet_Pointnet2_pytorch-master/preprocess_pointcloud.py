import os
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_preprocessed')

NUM_POINTS = 1024
VOXEL_SIZE = 0.05
SOR_NEIGHBORS = 10
SOR_STD_RATIO = 2.0

def load_pointcloud(file_path):
    try:
        points = np.loadtxt(file_path, delimiter=',').astype(np.float32)
    except:
        try:
            points = np.loadtxt(file_path).astype(np.float32)
        except:
            return None

    if points.ndim == 1 or points.shape[0] == 0:
        return None
    if points.shape[1] >= 3:
        points = points[:, :3]
    else:
        return None
    return points

# -------------------------------
# 纯 numpy 统计去噪（永不卡死）
# -------------------------------
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

# -------------------------------
# 纯 numpy 体素下采样（永不卡死）
# -------------------------------
def voxel_downsample(points):
    coords = np.floor(points / VOXEL_SIZE).astype(np.int32)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx]

# -------------------------------
# 主预处理
# -------------------------------
def preprocess_dataset():
    print("开始预处理数据集:", DATA_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    total_saved = 0
    total_skipped = 0

    for cls in os.listdir(DATA_DIR):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        out_cls_path = os.path.join(OUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)

        for file in tqdm(os.listdir(cls_path), desc=f"{cls}"):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(cls_path, file)
            points = load_pointcloud(file_path)

            if points is None or len(points) == 0:
                total_skipped += 1
                continue

            # 去噪 + 下采样（纯 numpy，永不崩溃）
            points = statistical_outlier_removal(points)
            points = voxel_downsample(points)

            # 统一采样到1024点
            if len(points) >= NUM_POINTS:
                choice = np.random.choice(len(points), NUM_POINTS, replace=False)
            else:
                choice = np.random.choice(len(points), NUM_POINTS, replace=True)
            points = points[choice, :]

            save_path = os.path.join(out_cls_path, file.replace('.txt', '.npy'))
            np.save(save_path, points)
            total_saved += 1

    print("✅ 预处理完成！")
    print("保存文件数:", total_saved)
    print("跳过文件数:", total_skipped)
    print("输出路径:", OUT_DIR)

if __name__ == "__main__":
    preprocess_dataset()