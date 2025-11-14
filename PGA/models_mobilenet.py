import torch
import torch.nn as nn
import torch.nn.functional as F

class GAP(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1)

class ProjHead(nn.Module):
    def __init__(self, in_c, out_c=512, hidden=None):
        super().__init__()
        hidden = hidden or out_c
        self.net = nn.Sequential(
            GAP(),
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
    def __init__(self, in_c=512, out_c=512, hidden=None):
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
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )
    def forward(self, x):
        return self.net(x)

class ConvBnPrelu(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
        )
    def forward(self, x):
        return self.net(x)

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3,3), stride=1, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, in_c, kernel=(1,1), stride=1, padding=0),
            ConvBnPrelu(in_c, in_c, kernel=kernel, stride=stride, padding=padding, groups=in_c),
            ConvBn(in_c, out_c, kernel=(1,1), stride=1, padding=0)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseRes(nn.Module):
    def __init__(self, in_c, out_c, kernel=(3,3), stride=1, padding=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding)
        self.downsample = (in_c != out_c or stride != 1)
        if self.downsample:
            self.shortcut = ConvBn(in_c, out_c, kernel=(1,1), stride=stride)
    def forward(self, x):
        if self.downsample:
            return self.net(x) + self.shortcut(x)
        else:
            return self.net(x) + x

class MultiDepthWiseRes(nn.Module):
    def __init__(self, num_block, channels, kernel=(3,3), stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding)
            for _ in range(num_block)
        ])
    def forward(self, x):
        return self.net(x)

class MobileNet(nn.Module):
    def __init__(self, proj_hidden=512, use_gem=False, use_pga=False):
        super().__init__()
        # ---- stem ----
        self.conv1 = ConvBnPrelu(3, 64, kernel=(3,3), stride=2, padding=1)
        self.conv2 = ConvBnPrelu(64, 64, kernel=(3,3), stride=1, padding=1)

        # ---- stages ----
        self.backbone1 = nn.Sequential(
            DepthWiseRes(64, 64, stride=1),
            MultiDepthWiseRes(num_block=2, channels=64)
        )  # ---- C = 64 ----
        self.backbone2 = nn.Sequential(
            DepthWiseRes(64, 128, stride=2),
            MultiDepthWiseRes(num_block=4, channels=128)
        )  # ---- C = 128 ----
        self.backbone3 = nn.Sequential(
            DepthWiseRes(128, 128, stride=1),
            MultiDepthWiseRes(num_block=4, channels=128)
        )  # ---- C = 128 ----
        self.backbone4 = nn.Sequential(
            DepthWiseRes(128, 256, stride=2),
            MultiDepthWiseRes(num_block=2, channels=256)
        )  # ---- C = 256 ----
        self.backbone5 = nn.Sequential(
            DepthWiseRes(256, 512, stride=2),
            MultiDepthWiseRes(num_block=2, channels=512)
        )  # ---- C = 512 ----

        # ---- 5 taps ----
        in_chs = [64, 128, 128, 256, 512]
        self.tap_proj = nn.ModuleList([
            ProjHead(c, out_c=512, hidden=proj_hidden) for c in in_chs
        ])

        self.final_proj = ProjHead1D(in_c=512, out_c=512, hidden=proj_hidden)

        # ---- 供 meta 标注: 每个 tap 的 conv 结束编号 ----
        self._tap_conv_end = [1, 2, 3, 4, 5]
        self._final_end = "gap"

        self.avgpool = GAP()

    # ---- 辅助: 跑一个 stage 并在每个 block 后取 tap ----
    def _run_block(self, x, block, tap_head=None):
        x = block(x)
        feat512 = tap_head(x) if (tap_head is not None) else None
        return x, feat512

    def forward(self, x):
        feats, meta = [], []

        # ---- stem ----
        x = self.conv1(x); x = self.conv2(x)

        # ---- 5 stages ----
        x, f = self._run_block(x, self.backbone1, self.tap_proj[0]); feats.append(f); meta.append({"name":"stage1","conv_end":self._tap_conv_end[0]})
        x, f = self._run_block(x, self.backbone2, self.tap_proj[1]); feats.append(f); meta.append({"name":"stage2","conv_end":self._tap_conv_end[1]})
        x, f = self._run_block(x, self.backbone3, self.tap_proj[2]); feats.append(f); meta.append({"name":"stage3","conv_end":self._tap_conv_end[2]})
        x, f = self._run_block(x, self.backbone4, self.tap_proj[3]); feats.append(f); meta.append({"name":"stage4","conv_end":self._tap_conv_end[3]})
        x, f = self._run_block(x, self.backbone5, self.tap_proj[4]); feats.append(f); meta.append({"name":"stage5","conv_end":self._tap_conv_end[4]})

        # ---- 分类输入: 512 向量 (GAP→Flatten) ----
        feat512 = torch.flatten(self.avgpool(x), 1)   # [B, 512]
        f_final = self.final_proj(feat512)            # [B, 512]
        feats.append(f_final)
        meta.append({"name":"final.gap512","conv_end":self._final_end})

        return feats, meta
