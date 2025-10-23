import torch
import torch.nn as nn
import torch.nn.functional as F
from models_mobilenet import MobileNet  # 这里必须改成你的文件名，比如 models_mobilenet_residual 或直接导入当前文件里的类

class MobileNetWithPGA(nn.Module):
    """
    这是在 MobileNet 外包一层的简单包装类
    功能：
      - 调用 MobileNet 得到各层特征 feats 和最终嵌入 emb
      - 对每个层的 [B,512] 特征做一次归一化（方便 PGA 计算相似度）
      - 输出：
          feats_512_list : [list of [B,512]]
          emb            : [B,512]
    """
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = MobileNet(embedding_size=embedding_size)  # 你的主网络
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(512) for _ in range(5)])  # 对每个层输出再归一化一次

    def forward(self, x):
        feats, emb = self.backbone(x)  # feats是list([B,512] * 5)
        feats_512_list = []

        # 对每一层的特征进行归一化（ArcFace/PGA都建议这样）
        for i, f in enumerate(feats):
            f = self.bn_list[i](f)
            f = F.normalize(f, dim=1)
            feats_512_list.append(f)

        return feats_512_list, emb
