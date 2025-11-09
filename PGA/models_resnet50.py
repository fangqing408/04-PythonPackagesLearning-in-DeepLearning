import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import os
os.environ['TORCH_HOME'] = './pre_ckpt' 

class GAP(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1)

class ProjHead(nn.Module):
    def __init__(self, in_c, out_c=512, hidden=None, use_gem=True):
        super().__init__()
        hidden = hidden or out_c
        self.net = nn.Sequential(
            GAP() if use_gem else nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.PReLU(hidden),
            nn.Linear(hidden, out_c, bias=False),
            nn.BatchNorm1d(out_c),
        )
    def forward(self, x):
        return self.net(x) # [B, out_c]
    
class ProjHead1D(nn.Module):
    def __init__(self, in_c=2048, out_c=512, hidden=None):  # [ADD]
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
    def __init__(self, proj_hidden=512, imagenet_init=True, use_gem=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None
        base = resnet50(weights=weights)

        # ---- stem ----
        self.conv1 = base.conv1 # [conv #1]
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # ---- layer1（3 blocks = conv #2..#10）----
        self.layer1_0 = base.layer1[0] # conv #2–#4
        self.layer1_1 = base.layer1[1] # conv #5–#7
        self.layer1_2 = base.layer1[2] # conv #8–#10

        # ---- layer2（4 blocks = conv #11..#22）----
        self.layer2_0 = base.layer2[0] # conv #11–#13
        self.layer2_1 = base.layer2[1] # conv #14–#16
        self.layer2_2 = base.layer2[2] # conv #17–#19
        self.layer2_3 = base.layer2[3] # conv #20–#22

        # ---- layer3（6 blocks = conv #23..#40）----
        self.layer3_0 = base.layer3[0] # conv #23–#25
        self.layer3_1 = base.layer3[1] # conv #26–#28
        self.layer3_2 = base.layer3[2] # conv #29–#31
        self.layer3_3 = base.layer3[3] # conv #32–#34
        self.layer3_4 = base.layer3[4] # conv #35–#37
        self.layer3_5 = base.layer3[5] # conv #38–#40

        # ---- layer4（3 blocks = conv #41..#49）----
        self.layer4_0 = base.layer4[0] # conv #41–#43
        self.layer4_1 = base.layer4[1] # conv #44–#46
        self.layer4_2 = base.layer4[2] # conv #47–#49

        self.avgpool = base.avgpool

        # ---- 9 个 tap 的投影头：layer3(C=1024)*6 + layer4(C=2048)*3 ----
        in_chs = [1024] * 6 + [2048] * 3
        self.tap_proj = nn.ModuleList([
            ProjHead(c, out_c=512, hidden=proj_hidden, use_gem=use_gem) for c in in_chs
        ])

        self.final_proj = ProjHead1D(in_c=2048, out_c=512, hidden=proj_hidden)

        # 供 meta 标注：每个 tap 的 conv 结束编号
        self._tap_conv_end = (
            [25, 28, 31, 34, 37, 40] +   # layer3.{0..5}
            [43, 46, 49]                  # layer4.{0..2}
        )

    # ---- 辅助：跑一个 stage 并在每个 block 后取 tap ----
    def _run_block(self, x, block, tap_head=None):
        x = block(x)
        feat512 = tap_head(x) if (tap_head is not None) else None
        return x, feat512

    def forward(self, x):
        feats, meta = [], []

        # stem
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)

        # layer1
        x, _ = self._run_block(x, self.layer1_0)
        x, _ = self._run_block(x, self.layer1_1)
        x, _ = self._run_block(x, self.layer1_2)

        # layer2
        x, _ = self._run_block(x, self.layer2_0)
        x, _ = self._run_block(x, self.layer2_1)
        x, _ = self._run_block(x, self.layer2_2)
        x, _ = self._run_block(x, self.layer2_3)

        # layer3（6 taps）
        x, f = self._run_block(x, self.layer3_0, self.tap_proj[0]); feats.append(f); meta.append({"name":"layer3.0","conv_end":25})
        x, f = self._run_block(x, self.layer3_1, self.tap_proj[1]); feats.append(f); meta.append({"name":"layer3.1","conv_end":28})
        x, f = self._run_block(x, self.layer3_2, self.tap_proj[2]); feats.append(f); meta.append({"name":"layer3.2","conv_end":31})
        x, f = self._run_block(x, self.layer3_3, self.tap_proj[3]); feats.append(f); meta.append({"name":"layer3.3","conv_end":34})
        x, f = self._run_block(x, self.layer3_4, self.tap_proj[4]); feats.append(f); meta.append({"name":"layer3.4","conv_end":37})
        x, f = self._run_block(x, self.layer3_5, self.tap_proj[5]); feats.append(f); meta.append({"name":"layer3.5","conv_end":40})

        # layer4（3 taps）
        x, f = self._run_block(x, self.layer4_0, self.tap_proj[6]); feats.append(f); meta.append({"name":"layer4.0","conv_end":43})
        x, f = self._run_block(x, self.layer4_1, self.tap_proj[7]); feats.append(f); meta.append({"name":"layer4.1","conv_end":46})
        x, f = self._run_block(x, self.layer4_2, self.tap_proj[8]); feats.append(f); meta.append({"name":"layer4.2","conv_end":49})

        # 分类输入：2048 向量（GAP→Flatten）
        feat2048 = torch.flatten(self.avgpool(x), 1)  # [B, 2048]
        f_final = self.final_proj(feat2048)           # [B, 512]
        feats.append(f_final)
        meta.append({"name":"final.gap2048","conv_end":"gap"})

        return feats, meta