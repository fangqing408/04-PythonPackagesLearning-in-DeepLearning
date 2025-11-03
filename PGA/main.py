from torch.utils.data import DataLoader
from torch import optim
from pga import PGAHead
from heads_arcface import ArcFaceHead
from pga_wrapper import MobileNetWithPGA
from dataset_overlap import OverlapSampler, OverlapDataset
from train_and_evaluate import train
from config import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_normal_(self.weight)
        # 放大角度差异，确保梯度强
        self.scale = scale
    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * (x @ w.t())
    
def reset_all(num_classes):
    # 重新初始化所有对象
    torch.cuda.empty_cache()
    backbone = MobileNetWithPGA(embedding_size=config.embedding_size).to(config.device)
    # softmax_head = nn.Linear(
    #     in_features=config.embedding_size, 
    #     out_features=num_classes
    # ).to(config.device) # pga 是必须的，其他的方法需要和他进行比较
    softmax_head = CosineClassifier(
        in_dim=config.embedding_size, 
        num_classes=num_classes, 
        scale=config.cosine_scale
    ).to(config.device)
    # 暂时先不对 arcface 分类头进行训练，先对比最基础的 softmax 分类头
    arcface_head = ArcFaceHead(in_features=config.embedding_size, out_features=num_classes).to(config.device)
    pga = PGAHead(
        num_layers=config.num_layers,
        device = config.device,
        topk=config.topk,
        t_diff=config.t_diff,
        use_ema = config.use_ema,
        ema_m = config.ema_m
    ).to(config.device)
    optimizer = optim.AdamW([
        {"params": backbone.parameters(), "lr": config.learning_rate},
        {"params": softmax_head.parameters(), "lr": config.learning_rate},
        {"params": pga.parameters(), "lr": config.learning_rate_pga}
    ], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=config.total_epochs,
        eta_min=config.learning_rate * 0.01
    )
    return backbone, softmax_head, pga, optimizer, scheduler

# ======================================================
# ===============创建 Overlap_DataLoader================
# ======================================================
if __name__ == "__main__":
    dataset = OverlapDataset(root=config.train_root, transform=config.train_transform)
    # 三天找不到原因，原来没打乱数据集，采样器直接顺序加载的导致每个 batch 里面最多只包含两个类别，导致每个 batch 里面的损失出现先降后升，最后一直降不下去的情况
    # 交叠了话，每个 batch 训练到的图是半静态的，怎么感觉越改越返璞归真
    sampler = OverlapSampler(
        data=dataset, 
        batch_size=config.batch_size, 
        overlap_ratio=config.overlap_ratio, 
        shuffle=config.overlap_shuffle
    )
    loader = DataLoader(
        dataset=dataset, 
        batch_size=config.batch_size, 
        shuffle=config.dataloader_shuffle, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory, 
        drop_last=config.drop_last
    )
    # imgs, labels = next(iter(loader))
    # print(labels) # 验证当前的 loader 加载的数据集的打乱情况，当前的 loader 纯手动实现的加载，防止出现没有交叠出现学不到结构的情况
    val_dataset = OverlapDataset(root=config.test_root, transform=config.test_transform)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory, 
        shuffle=config.dataloader_shuffle, 
        drop_last=config.drop_last
    )
    num_classes = dataset.num_classes
    backbone, softmax_head, pga, optimizer, scheduler = reset_all(num_classes=num_classes)
    train(
        name=config.tensorboard_name,
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
        lambda_K=config.lambda_K,
        lambda_Z=config.lambda_Z, 
        lambda_idea=config.lambda_idea,
        lambda_modify = config.lambda_modify
    )
# Dataset：MNIST（28×28 -> 128×128），未进行任何的图像增强，验证有效性
# ===========================================================================================================================================================================
# type_name | lambda_K | lambda_Z | lambda_idea | lambda_phase |    ema    | total_epochs | warmup_epochs |  alpha  | beta |  lr  | lr_pga | sigma_in | sigma_out | val_acc |
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#    pga_v1 |     4-64 |     4-16 |         1.0 |      0.5-0.5 | 0.85-0.95 |          100 |            10 | 1.0-1.2 |    \ | 1e-5 |   3e-6 |     0.99 |      0.00 |  98.60% |
#    pga_v2 | 4-64-6.4 | 4-16-1.6 |         1.0 |  0.5-0.2-0.3 | 0.85-0.95 |          100 |            10 | 1.0-1.2 |    \ | 1e-5 |   3e-6 |     0.99 |      0.00 |  98.61% |
#    pga_v3 |     4-64 |     4-16 |         1.0 |      0.5-0.5 |       0.9 |          100 |            10 | 1.0-1.2 |    \ | 1e-5 |   1e-6 |     0.99 |      0.00 |  98.66% |
#    pga_v4 |     4-64 |     4-16 |         1.0 |      0.5-0.5 |       0.9 |          100 |            10 | 1.0-1.2 |    \ | 1e-5 |   1e-6 |     0.99 |      0.00 |  98.63% |
#    pga_v5 |     4-64 |     4-16 |         1.0 |      0.5-0.5 |       0.9 |          100 |            10 | 1.0-1.2 |    \ | 1e-5 |   1e-6 |     0.99 |      0.00 |  98.66% |
#softmax_v1 |        \ |        \ |           \ |            \ |         \ |          100 |            10 |       \ |    \ | 1e-5 |      \ |        \ |         \ |  98.21% |
#softmax_v2 |        \ |        \ |           \ |            \ |         \ |          100 |            10 |       \ |    \ | 1e-5 |      \ |        \ |         \ |  98.17% |
# ===========================================================================================================================================================================

# MNIST 只有 10 类，特征维度低（28×28 灰度），类内差异很小，这意味着 batch 里每个类的样本有限，同类节点之间的相似度矩阵几乎是完美的块状结构
# 这也是为什么后期的损失下降缓慢的原因？
# 在这种场景下，普通的分类损失（CrossEntropy/ArcFace）已经能学到近乎最优的边界
# CASIA-WebFace 有上万类、几十万张人脸，每类人脸姿态、光照、年龄差异很大
# 同类样本在 embedding 空间中往往分散成多个子簇，所以在大规模高维人脸数据集上，PGA 的收益应该更明显且更稳定

# Dataset：CASIA-WebFace，进行了随机翻转 0.5，和（128 * 128 -> 144 * 144 -> 128 * 128）的随即裁剪，待做...