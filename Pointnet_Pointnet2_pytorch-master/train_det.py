import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import datetime
import os
from tqdm import tqdm

# 按照你的文件导入
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.pointnet_det import get_model

# ===== 参数 =====
BATCH_SIZE = 16
EPOCHS = 200
LR = 0.001
NUM_CLASS = 40          # ModelNet40
DATA_ROOT = 'data/modelnet40_normal_resampled'  # 你的数据集路径

# ===== 损失 =====
cls_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.MSELoss()

# ===== 日志配置：输出到 log/det/ 文件夹 + 控制台 =====
os.makedirs('log/det', exist_ok=True)
log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_name = f'log/det/train_{log_time}.log'

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# 文件日志
file_handler = logging.FileHandler(log_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def log_string(str):
    logger.info(str)
    print(str)

# ===== 精度计算 =====
def calculate_accuracy(pred_cls, label):
    pred_labels = torch.argmax(pred_cls, dim=1)
    correct = torch.sum(pred_labels == label).item()
    total = label.size(0)
    acc = correct / total
    return acc, correct, total, pred_labels

def calculate_class_acc(pred_labels, targets, num_class=40):
    class_correct = np.zeros(num_class)
    class_total = np.zeros(num_class)
    for t, p in zip(targets, pred_labels):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    class_acc = np.where(class_total == 0, 1.0, class_correct / class_total)
    mean_class_acc = np.mean(class_acc)
    return mean_class_acc

# ===== 测试 =====
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for points, label, bbox in dataloader:
            points = points.float().transpose(2, 1).to(device)
            label = label.long().to(device)
            bbox = bbox.float().to(device)

            pred_cls, _, _ = model(points)
            _, correct, total, preds = calculate_accuracy(pred_cls, label)
            total_correct += correct
            total_samples += total
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label.cpu().numpy())

    instance_acc = total_correct / total_samples
    class_acc = calculate_class_acc(all_preds, all_targets, NUM_CLASS)
    return instance_acc, class_acc

# ===== 训练 =====
def train():
    # 完全匹配你的 ModelNetDataLoader 需要的 args 参数
    class Args:
        num_point = 1024
        use_uniform_sample = False
        use_normals = False
        num_category = NUM_CLASS  # 必须加！你的数据集类需要这个参数

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_string(f"Using device: {device}")

    # ===================== 数据集加载（完全按你的 DataLoader 修正） =====================
    train_dataset = ModelNetDataLoader(
        root=DATA_ROOT,
        args=args,
        split='train',
        process_data=False  # 你的数据集类支持该参数
    )
    test_dataset = ModelNetDataLoader(
        root=DATA_ROOT,
        args=args,
        split='test',
        process_data=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型
    model = get_model(num_class=NUM_CLASS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_instance_acc = 0.0
    best_class_acc = 0.0

    log_string("=" * 50)
    log_string("Start training...")
    log_string("=" * 50)

    for epoch in range(EPOCHS):
        log_string(f"\n[ Epoch {epoch+1:2d} / {EPOCHS} ]")
        log_string("-" * 50)

        model.train()
        mean_correct = []

        # 训练批次
        for points, label, bbox in tqdm(train_loader, desc="Training"):
            points = points.float().transpose(2, 1).to(device)
            label = label.long().to(device)
            bbox = bbox.float().to(device)

            # 前向
            pred_cls, pred_bbox, _ = model(points)

            # 损失
            loss_cls = cls_loss_fn(pred_cls, label)
            loss_bbox = bbox_loss_fn(pred_bbox, bbox)
            loss = loss_cls + loss_bbox

            # 精度
            acc, correct, total, _ = calculate_accuracy(pred_cls, label)
            mean_correct.append(acc)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 训练集精度
        train_instance_acc = np.mean(mean_correct)

        # 测试集精度
        with torch.no_grad():
            test_instance_acc, class_acc = evaluate(model, test_loader, device)

            # 更新最佳
            if test_instance_acc > best_instance_acc:
                best_instance_acc = test_instance_acc
            if class_acc > best_class_acc:
                best_class_acc = class_acc

            # ===================== 你要求的每轮输出 =====================
            log_string(f"Train Instance Accuracy: {train_instance_acc:.6f}")
            log_string(f"Test Instance Accuracy : {test_instance_acc:.6f}")
            log_string(f"Class Accuracy        : {class_acc:.6f}")
            log_string(f"Best Instance Accuracy: {best_instance_acc:.6f}")
            log_string(f"Best Class Accuracy   : {best_class_acc:.6f}")

    log_string("\n" + "=" * 50)
    log_string("Training Finished!")
    log_string(f"Final Best Instance Acc: {best_instance_acc:.6f}")
    log_string(f"Final Best Class Acc   : {best_class_acc:.6f}")
    log_string("=" * 50)

if __name__ == '__main__':
    train()