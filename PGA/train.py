import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models_mobilenet import MobileNetBackbone
from models_pga_wrapper import MobileNetWithPGA
from heads_arcface import ArcFaceHead
from pga import PGAHead

# ---- 简单调度 ----
def linear_schedule(v0, v1, cur, total):
    t = min(cur/max(total,1), 1.0)
    return v0 + (v1 - v0) * t

def train_epoch_arcface(model, head, loader, optimizer, device):
    model.train(); head.train()
    ce = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, emb = model.forward_features(imgs)
        logits = head(emb, labels)
        loss = ce(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def train_epoch_pga(model_pga, pga_head, loader, optimizer, device,
                    epoch, total_epochs,
                    lambda_align=1.0, lambda_idea=1.0,
                    alpha_range=(1.0,1.8), beta_range=(1.0,0.6),
                    sigma_in_range=(0.8,0.98), sigma_out_range=(0.2,0.02),
                    warmup_epochs=2):
    model_pga.train(); pga_head.train()
    # 任务损失（演示用线性分类头；你可替换成度量损失/验证评估）
    ce = nn.CrossEntropyLoss()
    clf = None

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feats_512_list, emb = model_pga(imgs)     # 多层 512 + emb

        # α/β时间调度（全层共享）
        if epoch < warmup_epochs:
            alpha, beta = alpha_range[0], beta_range[0]
            lam_idea = 0.0
        else:
            e = epoch - warmup_epochs
            T = max(total_epochs - warmup_epochs, 1)
            alpha = linear_schedule(alpha_range[0], alpha_range[1], e, T)
            beta  = linear_schedule(beta_range[0],  beta_range[1],  e, T)
            lam_idea = lambda_idea
        with torch.no_grad():
            pga_head.alpha.fill_(alpha)
            pga_head.beta.fill_(beta)

        # PGA 损失
        Z_list, losses = pga_head(feats_512_list, labels,
                                  lambda_align=lambda_align, lambda_idea=lam_idea,
                                  sigma_in=linear_schedule(*sigma_in_range, epoch, total_epochs),
                                  sigma_out=linear_schedule(*sigma_out_range, epoch, total_epochs),
                                  stopgrad=True)

        # 简单监督：用最后一层Z做线性分类（演示）
        if clf is None:
            num_classes = int(labels.max().item())+1
            clf = nn.Linear(Z_list[-1].size(1), num_classes).to(device)
        logits = clf(Z_list[-1].detach())
        task_loss = ce(logits, labels)

        loss = task_loss + losses["loss_pga"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def main(run_mode="pga", epochs=5, device="cuda"):
    # TODO: 替换成你的 DataLoader
    # from your_dataset import train_loader
    raise NotImplementedError("把你的 train_loader 接进来 (imgs:[B,1,112,112], labels:[B]).")

    if run_mode == "arcface":
        model = MobileNetBackbone(embedding_size=512).to(device)
        arc = ArcFaceHead(in_features=512, out_features=num_classes, s=64.0, m=0.5).to(device)
        optim_all = optim.AdamW(list(model.parameters())+list(arc.parameters()), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            train_epoch_arcface(model, arc, train_loader, optim_all, device)

    elif run_mode == "pga":
        model_pga = MobileNetWithPGA(embedding_size=512, dims_in=[64,64,64,128,128,512]).to(device)
        pga_head  = PGAHead(num_layers=6, topk=16, t_diff=2, learnable_alpha_beta=False).to(device)
        optim_all = optim.AdamW(list(model_pga.parameters())+list(pga_head.parameters()), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            train_epoch_pga(model_pga, pga_head, train_loader, optim_all, device,
                            epoch=ep, total_epochs=epochs)

if __name__ == "__main__":
    pass
