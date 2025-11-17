import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 控制正负样本权重，>0.5 更重视负样本
        self.gamma = gamma  # 控制难易样本的聚焦程度
        self.reduction = reduction

    def forward(self, inputs, targets):
        eps = 1e-8
        inputs = torch.clamp(inputs, eps, 1. - eps)
        BCE = - (self.alpha * targets * torch.log(inputs) +
                 (1 - self.alpha) * (1 - targets) * torch.log(1 - inputs))
        focal = BCE * ((1 - inputs) ** self.gamma * targets +
                       (inputs ** self.gamma) * (1 - targets))
        if self.reduction == 'mean':
            return focal.mean()
        else:
            return focal.sum()
