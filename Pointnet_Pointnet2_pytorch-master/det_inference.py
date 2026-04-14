import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

# ===== 参数 =====
NUM_POINT = 1024
STRIDE = 0.5
BLOCK_SIZE = 1.0

CONF_THRESH = 0.8
IOU_THRESH = 0.3

TARGET_CLASSES = ['chair', 'table', 'sofa']

ROBUST_MODE = "rotate"   # clean / noise / dropout / rotate

# ===== 加载模型 =====
def load_model(model_path):
    from models.pointnet_det import get_model

    model = get_model(num_class=40)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model


# ===== 点云读取 =====
def load_pointcloud(file_path):
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    else:
        points = np.loadtxt(file_path, delimiter=',').astype(np.float32)

    return points[:, 0:3]


# ===== 切块 =====
def split_pointcloud(points):
    coord_min = np.min(points, axis=0)
    coord_max = np.max(points, axis=0)

    x_range = np.arange(coord_min[0], coord_max[0], STRIDE)
    y_range = np.arange(coord_min[1], coord_max[1], STRIDE)

    blocks = []

    for x in x_range:
        for y in y_range:
            cond = (
                (points[:, 0] >= x) & (points[:, 0] <= x + BLOCK_SIZE) &
                (points[:, 1] >= y) & (points[:, 1] <= y + BLOCK_SIZE)
            )

            block = points[cond]

            if len(block) < 50:
                continue

            choice = np.random.choice(len(block), NUM_POINT, replace=True)
            block = block[choice]

            blocks.append(block)

    return blocks


# ===== 归一化 =====
def normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    return points / m


# ===== IoU =====
def compute_iou(det1, det2):
    min1, max1 = det1[0], det1[1]
    min2, max2 = det2[0], det2[1]

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    inter = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter[0] * inter[1] * inter[2]

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    return inter_vol / (vol1 + vol2 - inter_vol + 1e-6)


# ===== NMS（带score排序）=====
def nms(detections, iou_thresh=0.3):
    if len(detections) == 0:
        return []

    # 按score排序
    detections = sorted(detections, key=lambda x: x[3], reverse=True)

    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)

        new_dets = []
        for det in detections:
            if det[2] != best[2]:
                new_dets.append(det)
                continue

            iou = compute_iou(best, det)
            if iou < iou_thresh:
                new_dets.append(det)

        detections = new_dets

    return keep


# ===== 预测 =====
def predict_blocks(model, blocks, shape_names):
    results = []

    for block in blocks:
        raw_block = block.copy()

        block = normalize(block)
        block = torch.tensor(block).unsqueeze(0).float()
        block = block.transpose(2, 1)

        if torch.cuda.is_available():
            block = block.cuda()

        pred_cls, pred_bbox, _ = model(block)

        prob = F.softmax(pred_cls, dim=1)
        score, cls_idx = torch.max(prob, dim=1)

        score = score.item()
        cls_idx = cls_idx.item()
        cls_name = shape_names[cls_idx]

        if score < CONF_THRESH:
            continue

        if cls_name not in TARGET_CLASSES:
            continue

        bbox = pred_bbox.cpu().detach().numpy()[0]
        cx, cy, cz, dx, dy, dz = bbox

        xyz_min = np.array([cx - dx/2, cy - dy/2, cz - dz/2])
        xyz_max = np.array([cx + dx/2, cy + dy/2, cz + dz/2])

        results.append((xyz_min, xyz_max, cls_name, score))

    return results


# ===== 可视化 =====
def draw_bbox(ax, min_pt, max_pt):
    x_min, y_min, z_min = min_pt
    x_max, y_max, z_max = max_pt

    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    for e in edges:
        p1, p2 = corners[e[0]], corners[e[1]]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]])


def visualize(points, detections):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], s=1)

    for det in detections:
        min_pt, max_pt, cls, score = det
        draw_bbox(ax, min_pt, max_pt)
        ax.text(min_pt[0], min_pt[1], min_pt[2], f"{cls}:{score:.2f}")

    plt.show()
# 鲁棒性测试
def add_noise(points, sigma=0.02):
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise


def random_dropout(points, drop_rate=0.3):
    mask = np.random.rand(len(points)) > drop_rate
    return points[mask]


def random_rotate(points):
    theta = np.random.uniform(0, 2*np.pi)

    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    return points @ rot
# ===== 主函数 =====
def main():
    model_path = 'det_model.pth'
    data_path = 'data/modelnet40_normal_resampled/chair/chair_0001.txt'

    shape_names = [line.strip() for line in open(
        'data/modelnet40_normal_resampled/modelnet40_shape_names.txt')]
    
    

    print("加载模型...")
    model = load_model(model_path)

    print("加载点云...")

    points = load_pointcloud(data_path)
    gt_min = np.min(points, axis=0)
    gt_max = np.max(points, axis=0)
    print("GT bbox:")
    print(gt_min, gt_max)

    print("原始点数:", len(points))

    if ROBUST_MODE == "noise":
        points = add_noise(points, sigma=0.02)
        print("加入高斯噪声")

    elif ROBUST_MODE == "dropout":
        points = random_dropout(points, drop_rate=0.3)
        print("随机丢点")

    elif ROBUST_MODE == "rotate":
        points = random_rotate(points)
        print("随机旋转")

    else:
        print("干净数据")

    print("切块...")
    blocks = split_pointcloud(points)
    print("block数量:", len(blocks))

    print("预测...")
    detections = predict_blocks(model, blocks, shape_names)

    print("NMS前数量:", len(detections))
    detections = nms(detections, IOU_THRESH)
    print("NMS后数量:", len(detections))

    print("检测结果:")
    for det in detections:
        print(det[2], det[3], det[0], det[1])

    print("\n===== 评估指标 =====")

    if len(detections) == 0:
        print("漏检（没有检测到目标）")
    else:
        for det in detections:
            pred_min, pred_max, cls, score = det

            iou = compute_iou(
                (pred_min, pred_max),
                (gt_min, gt_max)
            )

            print(f"类别: {cls}")
            print(f"置信度: {score:.4f}")
            print(f"IoU: {iou:.4f}")

    gt_center = (gt_min + gt_max) / 2
    pred_center = (pred_min + pred_max) / 2

    offset = np.linalg.norm(gt_center - pred_center)

    print(f"中心偏移: {offset:.4f}")

    print("可视化...")
    visualize(points, detections)


if __name__ == '__main__':
    main()