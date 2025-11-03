import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math, os
import torchvision.transforms as T
from PIL import Image

@torch.no_grad()
def evaluate(model_backbone, head, pairs_file, root, device):
    model_backbone.eval()
    head.eval()
    transform = T.Compose([
        # T.Grayscale(num_output_channels=1),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    sims, labels = [], []
    with open(pairs_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines):
        p1, p2, label = line.split()
        img1_path = os.path.join(root, p1)
        img2_path = os.path.join(root, p2)
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img1 = transform(img1).unsqueeze(0).to(device)
        img2 = transform(img2).unsqueeze(0).to(device)

        f1 = model_backbone(img1)[-1]
        f2 = model_backbone(img2)[-1]
        e1 = F.normalize(f1, dim=1)
        e2 = F.normalize(f2, dim=1)
        sims.append(F.cosine_similarity(e1, e2).cpu())
        labels.append(float(label))

    sims = torch.cat(sims)
    labels = torch.tensor(labels)

    best_acc, best_th = 0, 0
    for th in torch.linspace(0, 1, 50):
        acc = (((sims > th).float()) == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_th = acc, th.item()
    return best_th, best_acc

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
    
def train_CASIA(name, model_backbone, head, pga, loader, lfw_test_root, pairs_file, optimizer, scheduler, optimizer_pga, scheduler_pga, device, 
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
            total_loss.backward()
            optimizer.step()
            total_cls_loss += cls_loss.item()
            total_pga_loss += loss_pga.item()
            pbar.set_postfix({
                "cls": f"{cls_loss.item():.4f}",
                "pga": f"{loss_pga.item():.4f}"
            })
        best_th, best_acc = evaluate(
            model_backbone=model_backbone, 
            head=head, 
            pairs_file=pairs_file,
            root=lfw_test_root,
            device=device
        )
        avg_cls = total_cls_loss / len(loader)
        avg_pga = total_pga_loss / len(loader)
        writer.add_scalars("Loss/train", {
            "cls": avg_cls,
            "pga": avg_pga,
        }, epoch)
        writer.add_scalar("Acc/val", best_acc, epoch)
        lr = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        lr_pga = optimizer_pga.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | cls={avg_cls:.4f} | pga={avg_pga:.4f} | val_acc={best_acc * 100:.2f}% | best_th={best_th:.4f} | "
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
        torch.save(ckpt, f"./checkpoints/pga_casia_{epoch}.pth")
        scheduler.step()
        if epoch >= warmup_epochs:
            scheduler_pga.step()
    writer.close()