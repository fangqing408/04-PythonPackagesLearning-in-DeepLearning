import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        # ========== Step 1: 特征和权重归一化 ==========
        x = F.normalize(x, dim=1)                     # [B, D]
        W = F.normalize(self.weight, dim=1)           # [C, D]

        # ========== Step 2: 计算 cos(theta) ==========
        cos_theta = F.linear(x, W)                    # [B, C]
        cos_theta = cos_theta.clamp(-1.0, 1.0)        # 数值稳定性

        # ========== Step 3: 计算 cos(theta + m) ==========
        sin_theta = torch.sqrt(1.0 - cos_theta**2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # ========== Step 4: 处理边界情况 ==========
        # 避免角度超过 π - m，使用 piecewise 方式修正
        cond_mask = cos_theta - self.th
        cos_theta_m = torch.where(cond_mask > 0, cos_theta_m, cos_theta - self.mm)

        # ========== Step 5: 构建 one-hot 标签 ==========
        one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float()

        # ========== Step 6: 计算输出 logits ==========
        logits = (one_hot * cos_theta_m) + (1.0 - one_hot) * cos_theta
        logits = logits * self.s                      # 按 s 缩放

        return logits
