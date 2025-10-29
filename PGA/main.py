from torch.utils.data import DataLoader
from torch import optim
from pga import PGAHead
from models_mobilenet import MobileNet
from heads_arcface import ArcFaceHead
from pga_wrapper import MobileNetWithPGA
from overlap_dataset import OverlapSampler, OverlapDataset
from train_and_evaluate import train, evaluate
from config import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, in_dim=512, num_classes=10, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = scale  # 放大角度差异，确保梯度强

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)

        return self.scale * (x @ w.t())

def reset_all():
    torch.cuda.empty_cache()
    # 重新初始化所有对象
    backbone = MobileNetWithPGA(embedding_size=512).to(config.device)
    # softmax_head = nn.Linear(512, num_classes).to(config.device) # pga 是必须的，其他的方法需要和他进行比较
    softmax_head = CosineClassifier(in_dim=512, num_classes=10, scale=30.0).to(config.device)
    # arcface_head = ArcFaceHead(in_features=512, out_features=num_classes).to(config.device) # 暂时先不对 arcface 分类头进行训练，先对比最基础的 softmax 分类头
    pga = PGAHead(num_layers=5).to(config.device)

    optimizer = optim.AdamW([
        {"params": backbone.parameters(), "lr": config.learning_rate},
        {"params": softmax_head.parameters(), "lr": config.learning_rate},
        {"params": pga.parameters(), "lr": config.learning_rate * 0.1}
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.total_epochs,
        eta_min=config.learning_rate * 0.01
    )

    return backbone, softmax_head, pga, optimizer, scheduler

# ======================================================
# ===============创建 Overlap_DataLoader================
# ======================================================
if __name__ == "__main__":
    dataset = OverlapDataset(config.train_root, config.train_transform)
    sampler = OverlapSampler(dataset, batch_size=config.batch_size, overlap_ratio=config.overlap_ratio, shuffle=True)
    # 三天找不到原因，原来没打乱数据集，采样器直接顺序加载的导致每个 batch 里面最多只包含两个类别，导致每个 batch 里面的损失出现先降后升，最后一直降不下去的情况
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # 交叠了话，每个 batch 训练到的图是半静态的，怎么感觉越改越返璞归真
    # imgs, labels = next(iter(loader))
    # print(labels) # 验证当前的 loader 加载的数据集的打乱情况，当前的 loader 纯手动实现的加载，防止出现没有交叠出现学不到结构的情况
    val_dataset = OverlapDataset(config.test_root, config.test_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    num_classes = dataset.num_classes

    backbone, softmax_head, pga, optimizer, scheduler = reset_all()

    train(
        name="./log/pga_randomsampler_real",
        model_backbone=backbone,
        head=softmax_head,
        pga=pga,
        loader=loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.total_epochs,
        lambda_K=64,
        lambda_Z=16,
        lambda_idea=1.0
    )