import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder


class get_model(nn.Module):
    def __init__(self, num_class=40, normal_channel=False):
        super(get_model, self).__init__()

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)

        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_class)

        # bbox回归头
        self.fc_bbox1 = nn.Linear(1024, 512)
        self.bn_bbox1 = nn.BatchNorm1d(512)
        self.fc_bbox2 = nn.Linear(512, 256)
        self.bn_bbox2 = nn.BatchNorm1d(256)
        self.fc_bbox3 = nn.Linear(256, 6)  # cx,cy,cz,dx,dy,dz

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)

        # 分类
        cls = F.relu(self.bn1(self.fc1(x)))
        cls = F.relu(self.bn2(self.fc2(cls)))
        cls = self.fc3(cls)

        # 回归
        bbox = F.relu(self.bn_bbox1(self.fc_bbox1(x)))
        bbox = F.relu(self.bn_bbox2(self.fc_bbox2(bbox)))
        bbox = self.fc_bbox3(bbox)

        return cls, bbox, trans_feat