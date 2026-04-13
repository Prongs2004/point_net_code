import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.pointnet_det import get_model  # 你刚写的检测模型

# ===== 参数 =====
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001

# ===== 损失 =====
cls_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.MSELoss()

# ===== 训练 =====
def train():
    class Args:
        num_point = 1024
        use_uniform_sample = False
        use_normals = False
        num_category = 40

    args = Args()

    dataset = ModelNetDataLoader(
        root='data/modelnet40_preprocessed',
        args=args,
        split='train'
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model(num_class=40)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0

        for points, label, bbox in dataloader:

            points = points.float()
            label = label.long()
            bbox = bbox.float()

            points = points.transpose(2, 1)

            if torch.cuda.is_available():
                points = points.cuda()
                label = label.cuda()
                bbox = bbox.cuda()

            pred_cls, pred_bbox, _= model(points)

            loss_cls = cls_loss_fn(pred_cls, label)
            loss_bbox = bbox_loss_fn(pred_bbox, bbox)

            loss = loss_cls + loss_bbox

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'det_model.pth')


if __name__ == '__main__':
    train()