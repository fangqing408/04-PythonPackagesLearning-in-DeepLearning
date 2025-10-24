import torch.nn as nn
import torch.nn.functional as F
from models_mobilenet import MobileNet

class MobileNetWithPGA(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = MobileNet(embedding_size=embedding_size)
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(512) for _ in range(5)])

    def forward(self, x):
        feats, emb = self.backbone(x) # [B, 512] * 5
        feats_final = []

        for i, f in enumerate(feats):
            f = self.bn_list[i](f)
            f = F.normalize(f, dim=1)
            feats_final.append(f)
        return feats_final, emb # 这里没有再对 emb normalize
