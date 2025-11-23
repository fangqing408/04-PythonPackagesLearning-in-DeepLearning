# iresnet50_with_feats.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from iresnet import iresnet50


class ResNet50(nn.Module):
    def __init__(self, num_features=512, dropout=0, fp16=False):
        super().__init__()

        self.backbone = iresnet50(
            pretrained=False,
            num_features=num_features,
            dropout=dropout,
            fp16=fp16,
        )

    def _gap_flatten(self, x):
        # [B, C, H, W] -> [B, C]
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        feats, meta = [], []

        B = x.size(0)

        # ===== 输入像素当老师（和你之前风格保持一致） =====
        x_in = x.view(B, -1)
        feats.append(x_in)
        meta.append({"name": "input", "conv_end": "input"})

        fp16 = getattr(self.backbone, "fp16", False)

        # ===== 以下基本就是把 iresnet 原 forward 展开，只是中间多存了 GAP 特征 =====
        with torch.amp.autocast('cuda', enabled=fp16):
            # ---- stem ----
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.prelu(x)

            f_stem = self._gap_flatten(x)
            feats.append(f_stem)
            meta.append({"name": "stem", "conv_end": "stem"})

            # ---- layer1 ----
            x = self.backbone.layer1(x)
            f_l1 = self._gap_flatten(x)
            feats.append(f_l1)
            meta.append({"name": "layer1", "conv_end": 1})

            # ---- layer2 ----
            x = self.backbone.layer2(x)
            f_l2 = self._gap_flatten(x)
            feats.append(f_l2)
            meta.append({"name": "layer2", "conv_end": 2})

            # ---- layer3 ----
            x = self.backbone.layer3(x)
            f_l3 = self._gap_flatten(x)
            feats.append(f_l3)
            meta.append({"name": "layer3", "conv_end": 3})

            # ---- layer4 ----
            x = self.backbone.layer4(x)
            f_l4 = self._gap_flatten(x)
            feats.append(f_l4)
            meta.append({"name": "layer4", "conv_end": 4})

            # ---- 跟原版一样的结尾 ----
            x = self.backbone.bn2(x)
            x = torch.flatten(x, 1)
            x = self.backbone.dropout(x)

        # fc + features 放在 autocast 外面，这是官方写法
        x = self.backbone.fc(x.float() if fp16 else x)
        emb = self.backbone.features(x)   # [B, num_features]

        feats.append(emb)
        meta.append({"name": "final.emb", "conv_end": "gap"})

        return feats, meta
