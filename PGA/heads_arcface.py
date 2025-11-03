import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.5, enable_arcface=True):
        super().__init__()
        self.s = s
        self.m = m
        self.enable_arcface = enable_arcface
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, x, labels):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        cos_theta = F.linear(x, W)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        if self.enable_arcface:
            sin_theta = torch.sqrt(1.0 - cos_theta**2)
            cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
            cond_mask = cos_theta - self.th
            cos_theta_m = torch.where(cond_mask > 0, cos_theta_m, cos_theta - self.mm)
            one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float()
            logits = (one_hot * cos_theta_m) + (1.0 - one_hot) * cos_theta
        else:
            logits = cos_theta
        logits = logits * self.s
        return logits