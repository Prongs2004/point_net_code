import torch.nn as nn
import torch.nn.functional as F


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred_cls, pred_bbox, target_cls, target_bbox):
        # 分类损失
        cls_loss = F.cross_entropy(pred_cls, target_cls)

        # 回归损失（L2）
        bbox_loss = F.mse_loss(pred_bbox, target_bbox)

        # 总损失
        loss = cls_loss + 0.1 * bbox_loss

        return loss, cls_loss, bbox_loss