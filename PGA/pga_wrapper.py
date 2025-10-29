import torch.nn as nn
import torch.nn.functional as F
from models_mobilenet import MobileNet

class MobileNetWithPGA(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = MobileNet(embedding_size=embedding_size)
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(512) for _ in range(5)])

    def forward(self, x):
        feats = self.backbone(x)  # feats[-1] 已经是 backbone 输出
        feats_final = []

        for i, f in enumerate(feats):
            f = self.bn_list[i](f)
            f = F.normalize(f, dim=1)
            feats_final.append(f)
        return feats_final
