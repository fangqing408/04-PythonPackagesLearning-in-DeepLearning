from torch.utils.data import DataLoader
from torch import optim
from pga import PGAHead
from models_mobilenet import MobileNet
from heads_arcface import ArcFaceHead
from heads_pag import MobileNetWithPGA
from overlap_dataset import OverlapSampler, OverlapDataset
from train import train
from config import config
import torch
# ======================================================
# ===============创建 Overlap_DataLoader================
# ======================================================
if __name__ == "__main__":
    dataset = OverlapDataset(config.train_root, config.train_transform)
    sampler = OverlapSampler(dataset, batch_size=config.batch_size, overlap_ratio=config.overlap_ratio)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    num_classes = dataset.num_classes

    backbone = MobileNetWithPGA(embedding_size=512).to(config.device)
    arcface_head = ArcFaceHead(in_features=512, out_features=num_classes).to(config.device)
    pga_head = PGAHead(num_layers=5).to(config.device)
    # backbone.load_state_dict(torch.load("./checkpoints/ckpt_backbone_1.pth"))

    optimizer = optim.AdamW([
        {"params": backbone.parameters(), "lr": config.learning_rate},
        {"params": arcface_head.parameters(), "lr": config.learning_rate},
        {"params": pga_head.parameters(), "lr": config.learning_rate * 0.01}
    ], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.total_epochs,
        eta_min=config.learning_rate * 0.05
    )
    
    train(
        model_backbone=backbone,
        arcface_head=arcface_head,
        pga_head=pga_head,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.total_epochs,
        lambda_align_K=50,
        lambda_align_Z=50,
        lambda_idea=1.0
    )
