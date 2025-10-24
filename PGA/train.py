import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model_backbone, arcface_head, pga_head, loader, optimizer, scheduler, device, warmup_epochs=3, total_epochs=10, lambda_align_K=0.5, lambda_align_Z=0.5, lambda_idea=1.0):
    model_backbone.train()
    arcface_head.train()
    pga_head.train()

    for epoch in range(total_epochs):
        pbar = tqdm(loader, desc=f"[Epoch {epoch+1}/{total_epochs}]")
        total_cls_loss = 0.0
        total_pga_loss = 0.0

        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats_final, emb = model_backbone(imgs)
            logits = arcface_head(emb, labels)
            cls_loss = F.cross_entropy(logits, labels)

            if epoch >= warmup_epochs:
                losses = pga_head(
                    feats_final,
                    labels,
                    lambda_align_K=lambda_align_K,
                    lambda_align_Z=lambda_align_Z,
                    lambda_idea=lambda_idea
                )
                loss_pga = losses["loss_pga"]
            else:
                # warmup 阶段不训练 PGA，只训练分类器
                loss_pga = torch.zeros_like(cls_loss)

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

        avg_cls = total_cls_loss / len(loader)
        avg_pga = total_pga_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:03d} | Cls={avg_cls:.4f} | PGA={avg_pga:.4f} | LR={current_lr:.6f}")
        scheduler.step()
        

        torch.save(model_backbone.state_dict(), f"./checkpoints/_ckpt_backbone_{epoch}.pth")
        torch.save(pga_head.state_dict(), f"./checkpoints/_ckpt_pga_{epoch}.pth")