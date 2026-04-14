import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

from det_inference import (
    load_model,
    load_pointcloud,
    split_pointcloud,
    predict_blocks,
    nms,
    compute_iou
)

# ===== 参数 =====
MODEL_PATH = 'det_model.pth'
DATA_PATH = 'data/modelnet40_normal_resampled/chair/chair_0001.txt'

CONF_LIST_NOISE = [0.0, 0.01, 0.02, 0.05]
CONF_LIST_DROPOUT = [0.0, 0.2, 0.3, 0.5]
CONF_LIST_ROTATE = [0, 45, 90, 180]


# ===== 扰动函数 =====
def add_noise(points, sigma):
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise


def random_dropout(points, drop_rate):
    mask = np.random.rand(len(points)) > drop_rate
    return points[mask]


def rotate(points, angle_deg):
    theta = np.radians(angle_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return points @ rot


# ===== 单次评估 =====
def evaluate(model, points, shape_names):
    gt_min = np.min(points, axis=0)
    gt_max = np.max(points, axis=0)

    blocks = split_pointcloud(points)
    detections = predict_blocks(model, blocks, shape_names)
    detections = nms(detections)

    if len(detections) == 0:
        return 0, 0, 0  # 检测数, IoU, offset

    det = detections[0]
    pred_min, pred_max, cls, score = det

    iou = compute_iou((pred_min, pred_max), (gt_min, gt_max))

    center_gt = (gt_min + gt_max) / 2
    center_pred = (pred_min + pred_max) / 2
    offset = np.linalg.norm(center_gt - center_pred)

    return len(detections), iou, offset


# ===== 主流程 =====
def main():
    print("加载模型...")
    model = load_model(MODEL_PATH)

    shape_names = [line.strip() for line in open(
        'data/modelnet40_normal_resampled/modelnet40_shape_names.txt')]

    base_points = load_pointcloud(DATA_PATH)

    results = []

    # ===== 1. Noise =====
    print("\n=== Noise Test ===")
    for sigma in CONF_LIST_NOISE:
        pts = add_noise(base_points, sigma)
        det_num, iou, offset = evaluate(model, pts, shape_names)

        print(f"sigma={sigma}: IoU={iou:.3f}")

        results.append(["noise", sigma, det_num, iou, offset])

    # ===== 2. Dropout =====
    print("\n=== Dropout Test ===")
    for rate in CONF_LIST_DROPOUT:
        pts = random_dropout(base_points, rate)
        det_num, iou, offset = evaluate(model, pts, shape_names)

        print(f"drop={rate}: IoU={iou:.3f}")

        results.append(["dropout", rate, det_num, iou, offset])

    # ===== 3. Rotate =====
    print("\n=== Rotate Test ===")
    for angle in CONF_LIST_ROTATE:
        pts = rotate(base_points, angle)
        det_num, iou, offset = evaluate(model, pts, shape_names)

        print(f"angle={angle}: IoU={iou:.3f}")

        results.append(["rotate", angle, det_num, iou, offset])

    # ===== 保存CSV =====
    df = pd.DataFrame(results, columns=["type", "level", "det_num", "iou", "offset"])
    df.to_csv("results.csv", index=False)
    print("\n已保存 results.csv")

    # ===== 画图 =====
    plot_curve(df, "noise")
    plot_curve(df, "dropout")
    plot_curve(df, "rotate")


# ===== 画图 =====
def plot_curve(df, test_type):
    sub = df[df["type"] == test_type]

    plt.figure()
    plt.plot(sub["level"], sub["iou"], marker='o')
    plt.xlabel(test_type)
    plt.ylabel("IoU")
    plt.title(f"{test_type} vs IoU")
    plt.grid()

    filename = f"{test_type}_curve.png"
    plt.savefig(filename)
    print(f"已保存 {filename}")


if __name__ == "__main__":
    main()