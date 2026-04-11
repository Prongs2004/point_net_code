import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

# ====== 参数 ======
NUM_POINT = 1024
STRIDE = 0.5
BLOCK_SIZE = 1.0

TARGET_CLASSES = ['chair', 'table', 'person']

# ====== 加载模型 ======
def load_model(model_path, num_class=40):
    from models.pointnet_cls import get_model

    model = get_model(k=num_class, normal_channel=False)

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model


# ====== 点云读取 ======
def load_pointcloud(file_path):
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    else:
        points = np.loadtxt(file_path, delimiter=',').astype(np.float32)

    return points[:, 0:3]


# ====== 滑窗切块 ======
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


# ====== 归一化 ======
def normalize(points):
    points = points.astype(np.float32)

    centroid = np.mean(points, axis=0)
    points = points - centroid

    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / (m + 1e-6)

    return points


# ====== 预测 ======
def predict_blocks(model, blocks, classes):
    results = []

    for block in blocks:
        raw_block = block.copy()

        block = normalize(block)
        block = torch.tensor(block).unsqueeze(0).float()
        block = block.transpose(2, 1)

        if torch.cuda.is_available():
            block = block.cuda()

        pred, _ = model(block)
        pred = pred.argmax(dim=1).item()

        cls_name = classes[pred]

        if cls_name in TARGET_CLASSES:
            xyz_min = np.min(raw_block, axis=0)
            xyz_max = np.max(raw_block, axis=0)

            # ✅ 统一结构： (cls, min, max)
            results.append((cls_name, xyz_min, xyz_max))

    return results


# ====== 可视化（已修复） ======
def visualize(points, detections=None):
    print("开始绘图...")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.asarray(points).astype(float)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    if detections:
        for det in detections:
            cls, min_pt, max_pt = det

            min_pt = np.asarray(min_pt).astype(float)
            max_pt = np.asarray(max_pt).astype(float)

            # 8个角点
            corners = np.array([
                [min_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
            ])

            # 画角点
            ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=20)

            # 画边
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]

            for i, j in edges:
                ax.plot(
                    [corners[i][0], corners[j][0]],
                    [corners[i][1], corners[j][1]],
                    [corners[i][2], corners[j][2]]
                )

    plt.show()


def discover_input_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        return []

    valid_ext = ('.npy', '.txt', '.csv')
    files = []
    for root, _, filenames in os.walk(input_path):
        for name in filenames:
            if name.lower().endswith(valid_ext):
                files.append(os.path.join(root, name))

    return sorted(files)


# ====== 主函数 ======
def main():
    parser = argparse.ArgumentParser(description='Pseudo detection for point cloud files (single or batch).')
    parser.add_argument(
        '--model_path',
        default=os.path.join(BASE_DIR, 'log/classification/2026-04-04_21-11/checkpoints/best_model.pth'),
        help='Path to trained model checkpoint.'
    )
    parser.add_argument(
        '--input_path',
        default=os.path.join(BASE_DIR, 'data/modelnet40_normal_resampled/chair/chair_0001.txt'),
        help='Point cloud file path or directory path for batch detection.'
    )
    parser.add_argument(
        '--shape_names_path',
        default=os.path.join(BASE_DIR, 'data/modelnet40_normal_resampled/modelnet40_shape_names.txt'),
        help='Path to class names file.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize detection result. Recommended for single-file mode.'
    )
    args = parser.parse_args()

    shape_names = [line.strip() for line in open(args.shape_names_path)]
    input_files = discover_input_files(args.input_path)
    if len(input_files) == 0:
        print(f"未找到可检测文件: {args.input_path}")
        return

    print("加载模型...")
    model = load_model(args.model_path)

    total_detected_files = 0
    total_detections = 0
    single_mode = len(input_files) == 1

    for idx, file_path in enumerate(input_files, 1):
        print(f"\n[{idx}/{len(input_files)}] 检测文件: {file_path}")

        print("加载点云...")
        points = load_pointcloud(file_path)

        print("切块...")
        blocks = split_pointcloud(points)
        print(f"block数量: {len(blocks)}")

        print("预测...")
        detections = predict_blocks(model, blocks, shape_names)

        if detections:
            total_detected_files += 1
            total_detections += len(detections)

        print("检测结果:")
        if len(detections) == 0:
            print("无目标类别检测结果")
        else:
            for det in detections:
                print(det[0], det[1], det[2])

        if args.visualize and single_mode:
            print("可视化...")
            visualize(points, detections)

    print("\n===== 批量检测完成 =====")
    print(f"处理文件数: {len(input_files)}")
    print(f"有检测结果文件数: {total_detected_files}")
    print(f"总检测框数量: {total_detections}")


if __name__ == '__main__':
    main()
