import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
from config import config
from PIL import Image
import numpy as np
import os

@torch.no_grad()
def evaluate(model_backbone, head, pga, val_loader, device):
    model_backbone.eval()
    head.eval()
    pga.eval()

    correct, total = 0, 0
    total_loss = 0.0

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        feats_final = model_backbone(imgs)
        logits = head(feats_final[-1])

        total_loss += F.cross_entropy(logits, labels).item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(val_loader), correct / total

@torch.no_grad()
def evaluate_lfw(model_backbone, pga, device, pair_file=config.test_root_lfw_pair, img_root=config.test_root_lfw):
    model_backbone.eval()
    pga.eval()

    # 缓存所有图片特征
    feat_cache = {}

    def get_feat(img_path):
        if img_path in feat_cache:
            return feat_cache[img_path]
        img = Image.open(img_path).convert("L")
        img = config.test_transform(img).unsqueeze(dim=0).to(device)
        feats_final = model_backbone(img)
        feat = F.normalize(feats_final[-1], dim=1)
        feat_cache[img_path] = feat.cpu()
        return feat_cache[img_path]

    sims, labels = [], []

    with open(pair_file, "r") as f:
        lines = [x.strip().split() for x in f.readlines() if x.strip()]

    for path1, path2, lab in lines:
        img1 = os.path.join(img_root, path1)
        img2 = os.path.join(img_root, path2)

        f1 = get_feat(img1)
        f2 = get_feat(img2)
        sim = F.cosine_similarity(f1, f2).item()
        sims.append(sim)
        labels.append(int(lab))


    sims = np.array(sims)
    labels = np.array(labels)

    thresholds = np.linspace(0, 1, 50)
    accs = [(thr, ((sims > thr) == labels).mean()) for thr in thresholds]
    best_thr, best_acc = max(accs, key=lambda x: x[1])
    return best_acc, best_thr


def lambda_cosine(start, end, epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return start
    if epoch >= total_epochs:
        return end
    p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    p = max(0.0, min(p, 1.0))
    return end - (end - start) * (math.cos(math.pi * p) + 1.0) / 2.0

def train(name, model_backbone, head, pga, loader, val_loader, optimizer, scheduler, device, warmup_epochs=10, total_epochs=100, lambda_K=64, lambda_Z=16):
    writer = SummaryWriter(log_dir=name)
    # SB deepseek 给我说验证的时候 .eval() 了，让我把 .train() 放到 epoch 的循环里面导致某些量的累加被打断，损失下降极快但是正确率上不去，出现大问题，找了一晚上
    # MobileNet 有很多 BN 层，当每个 epoch 都重新进入 train 模式时，这些 BN 层的统计量积累被打断，训练不稳定，统计量频繁变化，可能过拟合，BN没有学到稳定的数据分布
    model_backbone.train()
    head.train()
    pga.train()

    for epoch in range(total_epochs):
        pbar = tqdm(loader, desc=f"[epoch {epoch}/{total_epochs - 1}]")
        total_cls_loss = 0.0
        total_pga_loss = 0.0
        
        lambda_align_K = lambda_cosine(4, lambda_K, epoch, total_epochs // 2, warmup_epochs)
        lambda_align_Z = lambda_cosine(4, lambda_Z, epoch, total_epochs // 2, warmup_epochs * 2)

        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats_final = model_backbone(imgs)

            if epoch >= warmup_epochs:
                losses = pga(feats_final, labels, lambda_align_K=lambda_align_K, lambda_align_Z=lambda_align_Z)
                loss_pga = losses["loss_pga"]
                logits = head(feats_final[-1])
            else: 
                # 用 feats_final[-1] 做分类，且不要影响 PGA 网络的梯度
                logits = head(feats_final[-1])
                loss_pga = torch.zeros(1, device=device)

            cls_loss = F.cross_entropy(logits, labels)

            total_loss = cls_loss + loss_pga
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_pga_loss += loss_pga.item()

            pbar.set_postfix({
                "cls": f"{cls_loss.item():.4f}",
                "pga": f"{loss_pga.item():.4f}"
            })

            # if iters % 1500 == 0:
            #     best_acc, best_thr = evaluate_lfw(model_backbone=model_backbone, pga=pga, device=device)

            #     avg_cls = total_cls_loss / len(loader)
            #     avg_pga = total_pga_loss / len(loader)

            #     writer.add_scalars("Loss/train", {
            #         "cls": avg_cls,
            #         "pga": avg_pga,
            #         # "val": val_avg_loss
            #     }, iters)
            #     writer.add_scalar("Acc/val", best_acc, iters)

        val_avg_loss, val_acc = evaluate(model_backbone, head, pga, val_loader, device)
        # best_acc, best_thr = evaluate_lfw(model_backbone=model_backbone, pga=pga, device=device)

        avg_cls = total_cls_loss / len(loader)
        avg_pga = total_pga_loss / len(loader)

        writer.add_scalars("Loss/train", {
            "cls": avg_cls,
            "pga": avg_pga,
            "val": val_avg_loss
        }, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        lr = optimizer.param_groups[0]["lr"]
        lr_pga = optimizer.param_groups[2]["lr"]
        print(
            f"epoch {epoch:03d} | cls={avg_cls:.4f} | pga={avg_pga:.4f} | val_loss={val_avg_loss:.4f} | "
            f"lr={lr:.8f} | lr_pga={lr_pga:.8f} | val_acc={val_acc * 100:.2f}% | lambda_align_K={lambda_align_K:.4f} | lambda_align_Z={lambda_align_Z:.4f}"
        )
        # print(
        #     f"epoch {epoch:03d} | cls={avg_cls:.4f} | pga={avg_pga:.4f} | best_thr: {best_thr:.3f} | best_acc: {best_acc*100:.2f}% | "
        #     f"lr={lr:.8f} | lr_pga={lr_pga:.8f} | lambda_align_K={lambda_align_K:.4f} | lambda_align_Z={lambda_align_Z:.4f}"
        # )
        ckpt = {
            "epoch": epoch,
            "model_backbone": model_backbone.state_dict(),
            "head": head.state_dict(),
            "pga": pga.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(ckpt, f"./checkpoints/pga_{epoch}.pth")
        scheduler.step()
    writer.close()