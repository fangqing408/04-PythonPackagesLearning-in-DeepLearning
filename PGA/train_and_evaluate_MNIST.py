import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import torchvision.transforms as T

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
    model_backbone.train()
    head.train()
    pga.train()
    return total_loss / len(val_loader), correct / total

def lambda_three_phase(epoch, total_epochs, warmup_epochs, peak_value, start_value=4, end_ratio=0.1, lambda_modify=False):
    rise_end = int(total_epochs * 0.5)
    if lambda_modify:
        flat_end = int(total_epochs * 0.7)
    else:
        flat_end = total_epochs
    end_value = peak_value * end_ratio
    if epoch < warmup_epochs:
        return start_value
    elif epoch < rise_end:
        t = (epoch - warmup_epochs) / max(rise_end - warmup_epochs, 1)
        return start_value + 0.5 * (1 - math.cos(math.pi * t)) * (peak_value - start_value)
    elif epoch < flat_end:
        return peak_value
    else:
        t = (epoch - flat_end) / max(total_epochs - flat_end, 1)
        return end_value + 0.5 * (1 + math.cos(math.pi * t)) * (peak_value - end_value)
    
def train_MNIST(name, model_backbone, head, pga, loader, val_loader, optimizer, scheduler, optimizer_pga, scheduler_pga, device, 
          warmup_epochs=10, total_epochs=100, lambda_K=64, lambda_Z=16, lambda_idea=1.0, lambda_modify=False):
    writer = SummaryWriter(log_dir=name)
    model_backbone.train()
    head.train()
    pga.train()
    for epoch in range(total_epochs):
        pbar = tqdm(loader, desc=f"[epoch {epoch}/{total_epochs - 1}]")
        total_cls_loss = 0.0
        total_pga_loss = 0.0
        lambda_align_K = lambda_three_phase(
            epoch=epoch, 
            total_epochs=total_epochs, 
            warmup_epochs=warmup_epochs, 
            peak_value=lambda_K, 
            start_value=4, 
            end_ratio=0.1, 
            lambda_modify=lambda_modify
        )
        lambda_align_Z = lambda_three_phase(
            epoch=epoch, 
            total_epochs=total_epochs, 
            warmup_epochs=warmup_epochs * 2, 
            peak_value=lambda_Z, 
            start_value=4, 
            end_ratio=0.1, 
            lambda_modify=lambda_modify
        )
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats_final = model_backbone(imgs)
            if epoch >= warmup_epochs:
                losses = pga(
                    feats_final=feats_final, 
                    labels=labels, 
                    lambda_align_K=lambda_align_K, 
                    lambda_align_Z=lambda_align_Z, 
                    lambda_idea=lambda_idea
                )
                loss_pga = losses["loss_pga"]
                logits = head(feats_final[-1])
            else: 
                # 用 feats_final[-1] 做分类，且不要影响 PGA 网络的梯度
                logits = head(feats_final[-1])
                loss_pga = torch.zeros(1, device=device)
            cls_loss = F.cross_entropy(logits, labels)
            total_loss = cls_loss + loss_pga
            optimizer.zero_grad()
            if epoch >= warmup_epochs:
                optimizer_pga.zero_grad()
            total_loss.backward()
            optimizer.step()
            if epoch >= warmup_epochs:
                optimizer_pga.step()
            total_cls_loss += cls_loss.item()
            total_pga_loss += loss_pga.item()
            pbar.set_postfix({
                "cls": f"{cls_loss.item():.4f}",
                "pga": f"{loss_pga.item():.4f}"
            })
        val_avg_loss, val_acc = evaluate(
            model_backbone=model_backbone, 
            head=head, 
            pga=pga, 
            val_loader=val_loader, 
            device=device
        )
        avg_cls = total_cls_loss / len(loader)
        avg_pga = total_pga_loss / len(loader)
        writer.add_scalars("Loss/train", {
            "cls": avg_cls,
            "pga": avg_pga,
            "val": val_avg_loss
        }, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        lr = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        lr_pga = optimizer_pga.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | cls={avg_cls:.4f} | pga={avg_pga:.4f} | val_avg_loss={val_avg_loss:.4f} | val_acc={val_acc * 100:.2f}% | "
            f"lr={lr:.8f} | lr_pga={lr_pga:.8f} | lr_head={lr_head:.8f} | lambda_align_K={lambda_align_K:.4f} | lambda_align_Z={lambda_align_Z:.4f}"
        )
        ckpt = {
            "epoch": epoch,
            "model_backbone": model_backbone.state_dict(),
            "head": head.state_dict(),
            "pga": pga.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_pga": optimizer_pga.state_dict(),
            "scheduler_pga": scheduler_pga.state_dict()
        }
        torch.save(ckpt, f"./checkpoints/pga_{epoch}.pth")
        scheduler.step()
        if epoch >= warmup_epochs:
            scheduler_pga.step()
    writer.close()