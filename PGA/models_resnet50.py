import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import os
os.environ['TORCH_HOME'] = './pre_ckpt' 

class ProjHead1D(nn.Module):
    def __init__(self, in_c=2048, out_c=512, hidden=None):
        super().__init__()
        hidden = hidden or out_c
        self.net = nn.Sequential(
            nn.Linear(in_c, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.PReLU(hidden),
            nn.Linear(hidden, out_c, bias=False),
            nn.BatchNorm1d(out_c),
        )

    def forward(self, x):
        return self.net(x)


class ResNet50(nn.Module):
    def __init__(self, proj_hidden=512, imagenet_init=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None
        base = resnet50(weights=weights)

        # ---- stem ----
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # ---- layer1（3 blocks）----
        self.layer1_0 = base.layer1[0]
        self.layer1_1 = base.layer1[1]
        self.layer1_2 = base.layer1[2]

        # ---- layer2（4 blocks）----
        self.layer2_0 = base.layer2[0]
        self.layer2_1 = base.layer2[1]
        self.layer2_2 = base.layer2[2]
        self.layer2_3 = base.layer2[3]

        # ---- layer3（6 blocks）----
        self.layer3_0 = base.layer3[0]
        self.layer3_1 = base.layer3[1]
        self.layer3_2 = base.layer3[2]
        self.layer3_3 = base.layer3[3]
        self.layer3_4 = base.layer3[4]
        self.layer3_5 = base.layer3[5]

        # ---- layer4（3 blocks）----
        self.layer4_0 = base.layer4[0]
        self.layer4_1 = base.layer4[1]
        self.layer4_2 = base.layer4[2]

        self.avgpool = base.avgpool   # 最后一个 GAP

        # 最终投影：2048 -> 512 给 ArcFace / 分类头
        self.final_proj = ProjHead1D(in_c=2048, out_c=512, hidden=proj_hidden)

    def _run_block(self, x, block):
        x = block(x)
        return x

    def _gap_flatten(self, x):
        # 无参数 readout，直接 GAP + flatten
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        feats, meta = [], []

        B = x.size(0)
        # 展平输入，变成 [B, 3*112*112] 的向量
        x_in = x.view(B, -1).detach()   # detach 可要可不要，这里加上更明确“当老师”
        feats.append(x_in)
        meta.append({"name": "input", "conv_end": "input"})

        # ===== stem =====
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 老师：stem 后的 feature，维度大约是 C=64
        teacher_feat = self._gap_flatten(x)          # [B, 64]
        feats.append(teacher_feat)
        meta.append({"name": "stem.teacher", "conv_end": "stem"})

        # ===== layer1 =====
        x = self._run_block(x, self.layer1_0)
        x = self._run_block(x, self.layer1_1)
        x = self._run_block(x, self.layer1_2)
        feat_l1 = self._gap_flatten(x)               # [B, 256]
        feats.append(feat_l1)
        meta.append({"name": "layer1", "conv_end": 1})

        # ===== layer2 =====
        x = self._run_block(x, self.layer2_0)
        x = self._run_block(x, self.layer2_1)
        x = self._run_block(x, self.layer2_2)
        x = self._run_block(x, self.layer2_3)
        feat_l2 = self._gap_flatten(x)               # [B, 512]
        feats.append(feat_l2)
        meta.append({"name": "layer2", "conv_end": 2})

        # ===== layer3 =====
        x = self._run_block(x, self.layer3_0)
        x = self._run_block(x, self.layer3_1)
        x = self._run_block(x, self.layer3_2)
        x = self._run_block(x, self.layer3_3)
        x = self._run_block(x, self.layer3_4)
        x = self._run_block(x, self.layer3_5)
        feat_l3 = self._gap_flatten(x)               # [B, 1024]
        feats.append(feat_l3)
        meta.append({"name": "layer3", "conv_end": 3})

        # ===== layer4 + final =====
        x = self._run_block(x, self.layer4_0)
        x = self._run_block(x, self.layer4_1)
        x = self._run_block(x, self.layer4_2)

        feat2048 = torch.flatten(self.avgpool(x), 1)  # [B, 2048]
        f_final = self.final_proj(feat2048)           # [B, 512]
        feats.append(f_final)
        meta.append({"name": "final.gap2048", "conv_end": "gap"})

        return feats, meta
