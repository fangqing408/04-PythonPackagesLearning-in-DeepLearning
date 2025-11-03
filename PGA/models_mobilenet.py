import torch.nn as nn

# ======================================================
# ====================基础模块==========================
# ======================================================
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
    
# ======================================================
# ===================骨架网络===========================
# ======================================================
class MobileNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.conv1 = ConvBnPrelu(1, 64, kernel=(3,3), stride=2, padding=1)
        self.conv2 = ConvBnPrelu(64, 64, kernel=(3,3), stride=1, padding=1)
        self.backbone1 = nn.Sequential(
            DepthWiseRes(64, 64, stride=1),
            MultiDepthWiseRes(num_block=2, channels=64)
        )
        self.backbone2 = nn.Sequential(
            DepthWiseRes(64, 128, stride=2),
            MultiDepthWiseRes(num_block=4, channels=128)
        )
        self.backbone3 = nn.Sequential(
            DepthWiseRes(128, 128, stride=1),
            MultiDepthWiseRes(num_block=4, channels=128)
        )
        self.backbone4 = nn.Sequential(
            DepthWiseRes(128, 256, stride=2),
            MultiDepthWiseRes(num_block=2, channels=256)
        )
        self.backbone5 = nn.Sequential(
            DepthWiseRes(256, 512, stride=2),
            MultiDepthWiseRes(num_block=2, channels=512)
        )
        # ==============================================
        # ===投影将中间骨架是输出投影到 512 维空间中的点===
        # ==============================================
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(c, 512, bias=False),
                nn.BatchNorm1d(512)
            )
            for c in [64, 128, 128, 256, 512]
        ])
        self.fc = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        feats = []  # 每个 backbone 输出的[B, 512]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.backbone1(x)
        feats.append(self.proj[0](x))
        x = self.backbone2(x)
        feats.append(self.proj[1](x))
        x = self.backbone3(x)
        feats.append(self.proj[2](x))
        x = self.backbone4(x)
        feats.append(self.proj[3](x))
        x = self.backbone5(x)
        feats.append(self.proj[4](x))
        return feats