import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math, os
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils import clip_grad_norm_ 

@torch.no_grad()
def evaluate(model, head, pairs_file, root, device):
    model.eval()
    head.eval()
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    sims, labels = [], []
    with open(pairs_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc="LFW eval"):
        p1, p2, label = line.split()
        img1_path = os.path.join(root, p1)
        img2_path = os.path.join(root, p2)
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img1 = transform(img1).unsqueeze(0).to(device)
        img2 = transform(img2).unsqueeze(0).to(device)

        feats1, _ = model(img1)
        feats2, _ = model(img2)
        e1 = F.normalize(feats1[-1], dim=1, eps=1e-8)
        e2 = F.normalize(feats2[-1], dim=1, eps=1e-8)
        sims.append(F.cosine_similarity(e1, e2).cpu())
        labels.append(float(label))

    sims = torch.cat(sims)
    labels = torch.tensor(labels)

    best_acc, best_th = 0, 0
    for th in torch.linspace(-1, 1, 10000):
        acc = (((sims > th).float()) == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_th = acc, th.item()

    model.train()
    head.train()
    return best_th, best_acc

def lambda_two_phase(epoch, total_epochs, warmup_epochs,
                     peak_value, start_value=0.1):
    rise_end = int(total_epochs * 0.5)
    if epoch < warmup_epochs:
        return start_value

    if epoch < rise_end:
        t = (epoch - warmup_epochs) / max(rise_end - warmup_epochs, 1)
        coef = 0.5 * (1 - math.cos(math.pi * t))  # 0~1
        return start_value + coef * (peak_value - start_value)

    return peak_value

def train_CASIA_ResNet50(
    name,
    model,
    head,
    pga,
    loader,
    lfw_test_root,
    pairs_file,
    optimizer,
    scheduler,
    optimizer_pga,
    scheduler_pga,
    device,
    warmup_epochs=8,
    total_epochs=32,
    lambda_K=1.0,
):
    writer = SummaryWriter(log_dir=name)
    max_grad_norm = 30.0

    for epoch in range(total_epochs):
        model.train()
        pga.train()

        # 这一整个 epoch 用同一个 lambda_align_K
        lambda_align_K = lambda_two_phase(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            peak_value=lambda_K,
            start_value=0.1 * lambda_K,
        )

        pbar = tqdm(loader, desc=f"[epoch {epoch}/{total_epochs - 1}]")

        # 纯 epoch 统计
        epoch_cls_loss = 0.0
        epoch_pga_loss = 0.0
        epoch_rawpga_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_clip_cnt = 0
        epoch_step_cnt = 0
        last_debug_pga = None

        for imgs, labels, sample_ids in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_ids = sample_ids.to(device, non_blocking=True)

            feats, meta = model(imgs)

            # PGA 部分：warmup 之后才启用
            if epoch >= warmup_epochs:
                losses = pga(
                    feats_final=feats,
                    labels=labels,
                    sample_ids=sample_ids,
                    lambda_align_K=lambda_align_K
                )
                loss_pga = losses["loss_pga"]
                raw_pga = losses["raw_pga"]
                last_debug_pga = losses["debug"]
            else:
                loss_pga = torch.zeros(1, device=device)
                raw_pga = torch.zeros(1, device=device)
                last_debug_pga = None

            # ArcFace 分类头
            logits = head(feats[-1], labels, True)
            cls_loss = F.cross_entropy(logits, labels)
            total_loss = cls_loss + loss_pga

            optimizer.zero_grad()
            if epoch >= warmup_epochs:
                optimizer_pga.zero_grad()

            total_loss.backward()

            # 梯度裁剪
            params = list(model.parameters()) + list(head.parameters())
            if epoch >= warmup_epochs:
                params += list(pga.parameters())
            grad_norm = clip_grad_norm_(params, max_norm=max_grad_norm)
            is_clipped = float(grad_norm > max_grad_norm)

            optimizer.step()
            if epoch >= warmup_epochs:
                optimizer_pga.step()

            # epoch 级别统计
            epoch_cls_loss += cls_loss.item()
            epoch_pga_loss += loss_pga.item()
            epoch_rawpga_loss += raw_pga.item()
            epoch_grad_norm += float(grad_norm)
            epoch_clip_cnt += int(is_clipped)
            epoch_step_cnt += 1

            pbar.set_postfix({
                "cls": f"{cls_loss.item():.4f}",
                "pga": f"{loss_pga.item():.4f}",
                "raw_pga": f"{raw_pga.item():.4f}",
                "grad": f"{float(grad_norm):.2f}",
                "clip": int(is_clipped),
            })

        # ===== 一个 epoch 结束：调 lr（纯 epoch 控制）=====
        scheduler.step()
        if epoch >= warmup_epochs:
            scheduler_pga.step()

        # ===== 一个 epoch 结束：做一次 LFW 验证 =====
        best_th, best_acc = evaluate(
            model=model,
            head=head,
            pairs_file=pairs_file,
            root=lfw_test_root,
            device=device
        )

        # ===== 计算 epoch 平均指标 =====
        avg_cls = epoch_cls_loss / max(epoch_step_cnt, 1)
        avg_pga = epoch_pga_loss / max(epoch_step_cnt, 1)
        avg_rawpga = epoch_rawpga_loss / max(epoch_step_cnt, 1)
        avg_grad = epoch_grad_norm / max(epoch_step_cnt, 1)
        clip_ratio = epoch_clip_cnt / max(epoch_step_cnt, 1)

        # ===== TensorBoard：用 epoch 当 step =====
        writer.add_scalars("Loss/train", {
            "cls": avg_cls,
            "pga": avg_pga,
            "rawpga": avg_rawpga,
            "grad": avg_grad,
        }, epoch)
        writer.add_scalar("Acc/val", best_acc, epoch)
        writer.add_scalar("LR/main", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/pga", optimizer_pga.param_groups[0]["lr"], epoch)
        writer.add_scalar("Lambda/align_K", lambda_align_K, epoch)

        if epoch >= warmup_epochs and last_debug_pga is not None:
            writer.add_scalars("PGA/subgraph", {
                "b_sub": last_debug_pga["b_sub"],
                "sub_ratio": last_debug_pga["sub_ratio"],
                "eff_edges_mean": last_debug_pga["eff_edges_mean"],
            }, epoch)

            mr0 = last_debug_pga["main_ratio_per_layer"][0]
            mr_last = last_debug_pga["main_ratio_per_layer"][-1]
            writer.add_scalars("PGA/main_ratio", {
                "layer0": mr0,
                "layer_last": mr_last,
            }, epoch)

        # ===== 打一行 summary log =====
        lr_main = optimizer.param_groups[0]["lr"]
        lr_pga = optimizer_pga.param_groups[0]["lr"]
        print(
            f"[epoch {epoch:03d}] "
            f"cls={avg_cls:.4f} | pga={avg_pga:.4f} | rawpga={avg_rawpga:.4f}"
            f"| val_acc={best_acc * 100:.2f}% | best_th={best_th:.4f} "
            f"| lr={lr_main:.6f} | lr_pga={lr_pga:.6f} "
            f"| lambda_K={lambda_align_K:.4f} "
            f"| clip_ratio={clip_ratio:.3f} | grad={avg_grad:.2f}"
        )

        # ===== 按 epoch 存 ckpt =====
        ckpt = {
            "epoch": epoch,
            "model_backbone": model.state_dict(),
            "head": head.state_dict(),
            "pga": pga.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_pga": optimizer_pga.state_dict(),
            "scheduler_pga": scheduler_pga.state_dict(),
        }
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(ckpt, f"./checkpoints/pga_casia_{epoch:03d}.pth")

    writer.close()
