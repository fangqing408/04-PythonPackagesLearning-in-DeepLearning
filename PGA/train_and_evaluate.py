import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math

@torch.no_grad()
def evaluate(model_backbone, head, pga, val_loader, device, epoch, warmup_epochs):
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


def lambda_cosine(start, end, epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return start
    if epoch >= total_epochs:
        return end
    p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    p = max(0.0, min(p, 1.0))
    return end - (end - start) * (math.cos(math.pi * p) + 1.0) / 2.0

def train(name, model_backbone, head, pga, loader, val_loader, optimizer, scheduler, device, warmup_epochs=5, total_epochs=40, lambda_K=64, lambda_Z=32, lambda_idea=1.0):
    writer = SummaryWriter(log_dir=name)
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
                losses = pga(feats_final, labels, lambda_align_K=lambda_align_K, lambda_align_Z=lambda_align_Z, lambda_idea=lambda_idea)
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

        val_avg_loss, val_acc = evaluate(model_backbone, head, pga, val_loader, device, epoch, warmup_epochs)

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