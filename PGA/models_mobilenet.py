import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 基础模块 ==========
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


# ========== 残差模块 ==========
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


# ========== 新版 MobileNet（结构清晰、每个block输出[B,512]） ==========
class MobileNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()

        # ---- stem ----
        self.conv1 = ConvBnPrelu(1, 64, kernel=(3,3), stride=2, padding=1)
        self.conv2 = ConvBnPrelu(64, 64, kernel=(3,3), stride=1, padding=1)

        # ---- backbone1~5 ----
        self.backbone1 = nn.Sequential(
            DepthWiseRes(64, 64, stride=1),
            DepthWiseRes(64, 64, stride=1)
        )
        self.backbone2 = nn.Sequential(
            DepthWiseRes(64, 128, stride=2),
            MultiDepthWiseRes(num_block=2, channels=128)
        )
        self.backbone3 = nn.Sequential(
            DepthWiseRes(128, 128, stride=1),
            MultiDepthWiseRes(num_block=3, channels=128)
        )
        self.backbone4 = nn.Sequential(
            DepthWiseRes(128, 256, stride=2),
            MultiDepthWiseRes(num_block=3, channels=256)
        )
        self.backbone5 = nn.Sequential(
            DepthWiseRes(256, 512, stride=2),
            MultiDepthWiseRes(num_block=2, channels=512)
        )

        # ---- 统一到 512维 ----
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), # 不关心样本内的结构，所以这个还是比较好的
                Flatten(),
                nn.Linear(c, 512, bias=False),
                nn.BatchNorm1d(512)
            )
            for c in [64, 128, 128, 256, 512]
        ])

        # ---- 最终 embedding 层 ----
        self.fc = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)


    def forward(self, x):
        feats = []  # 每个backbone输出的[B,512]
        x = self.conv1(x)
        x = self.conv2(x)

        # backbone1
        x = self.backbone1(x)
        feats.append(self.proj[0](x))

        # backbone2
        x = self.backbone2(x)
        feats.append(self.proj[1](x))

        # backbone3
        x = self.backbone3(x)
        feats.append(self.proj[2](x))

        # backbone4
        x = self.backbone4(x)
        feats.append(self.proj[3](x))

        # backbone5
        x = self.backbone5(x)
        feats.append(self.proj[4](x))

        # 最终 embedding
        emb = self.bn(self.fc(feats[-1]))

        return feats, emb
