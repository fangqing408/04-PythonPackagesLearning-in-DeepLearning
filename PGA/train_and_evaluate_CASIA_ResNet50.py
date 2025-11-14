import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math, os
import torchvision.transforms as T
from PIL import Image
from PIL import ImageFile
from torch.nn.utils import clip_grad_norm_ 

@torch.no_grad()
def evaluate(model, head, pairs_file, root, device):
    model.eval()
    head.eval()
    transform = T.Compose([
        # T.Grayscale(num_output_channels=1),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        feats1, _ = model(img1)
        feats2, _ = model(img2)
        e1 = F.normalize(feats1[-1], dim=1, eps=1e-8)
        e2 = F.normalize(feats2[-1], dim=1, eps=1e-8)
        sims.append(F.cosine_similarity(e1, e2).cpu())
        labels.append(float(label))

    sims = torch.cat(sims)
    labels = torch.tensor(labels)

    best_acc, best_th = 0, 0
    for th in torch.linspace(-1, 1, 2000):
        acc = (((sims > th).float()) == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_th = acc, th.item()
    model.train()
    head.train()
    return best_th, best_acc

def lambda_three_phase(epoch, total_epochs, warmup_epochs, peak_value, start_value=4, end_ratio=0.1, lambda_modify=False):
    rise_end = int(total_epochs * 0.5)
    if lambda_modify: flat_end = int(total_epochs * 0.7)
    else: flat_end = total_epochs
    end_value = peak_value * end_ratio
    if epoch < warmup_epochs: return start_value
    elif epoch < rise_end:
        t = (epoch - warmup_epochs) / max(rise_end - warmup_epochs, 1)
        return start_value + 0.5 * (1 - math.cos(math.pi * t)) * (peak_value - start_value)
    elif epoch < flat_end: return peak_value
    else:
        t = (epoch - flat_end) / max(total_epochs - flat_end, 1)
        return end_value + 0.5 * (1 + math.cos(math.pi * t)) * (peak_value - end_value)
    
def train_CASIA_ResNet50(name, model, head, pga, loader, lfw_test_root, pairs_file, optimizer, scheduler, optimizer_pga, scheduler_pga, device, 
          warmup_epochs=10, total_epochs=100, lambda_K=64, lambda_Z=16, lambda_modify=False):
    writer = SummaryWriter(log_dir=name)
    model.train()
    pga.train()
    # iters 统计 tensorboard
    step = 0
    log_step = 965
    win_cls = win_pga = win_grad = 0
    win_clip = win_step = 0
    win_rawk = win_rawz = win_rawpga = 0

    max_grad_norm = 15.0 # 本步总梯度大小超过 8，整体缩回到等效大小为 8
    for epoch in range(total_epochs):
        pbar = tqdm(loader, desc=f"[epoch {epoch}/{total_epochs - 1}]")
        total_cls_loss = 0.0
        total_pga_loss = 0.0
        lambda_align_K = lambda_three_phase(epoch=epoch, total_epochs=total_epochs, warmup_epochs=warmup_epochs, 
            peak_value=lambda_K, start_value=0, end_ratio=0.1, lambda_modify=lambda_modify
        )
        lambda_align_Z = lambda_three_phase(epoch=epoch, total_epochs=total_epochs, warmup_epochs=warmup_epochs * 2, 
            peak_value=lambda_Z, start_value=0, end_ratio=0.1, lambda_modify=lambda_modify)
        clipped_cnt = 0
        grad_norm_sum = 0.0
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats, meta = model(imgs)
            if epoch >= warmup_epochs:
                losses = pga(feats_final=feats, labels=labels, lambda_align_K=lambda_align_K, lambda_align_Z=lambda_align_Z, )
                loss_pga = losses["loss_pga"]
                raw_alignK = float(losses["loss_align_K"].item())
                raw_alignZ = float(losses["loss_align_Z"].item())
                raw_pga = raw_alignK + raw_alignZ

                win_rawk += raw_alignK
                win_rawz += raw_alignZ
                win_rawpga += raw_pga
                logits = head(feats[-1], labels, False)
            else: 
                logits = head(feats[-1], labels, False)
                loss_pga = torch.zeros(1, device=device)
            cls_loss = F.cross_entropy(logits, labels)
            total_loss = cls_loss + loss_pga
            optimizer.zero_grad()
            if epoch >= warmup_epochs:
                optimizer_pga.zero_grad()
            total_loss.backward()

            params = list(model.parameters()) + list(head.parameters())
            if epoch >= warmup_epochs:
                params += list(pga.parameters())
            # 把所有的参数的梯度拼成一个很长的向量，计算这个向量的长度，大于 max_norm，就整体的缩放，返回缩放前的长度
            grad_norm = clip_grad_norm_(params, max_norm=max_grad_norm)
            is_clipped = float(grad_norm > max_grad_norm)
            clipped_cnt += int(is_clipped)
            grad_norm_sum += float(grad_norm)

            total_cls_loss += cls_loss.item()
            total_pga_loss += loss_pga.item()
            post = {"cls": f"{cls_loss.item():.4f}","pga": f"{loss_pga.item():.4f}","grad": f"{float(grad_norm):.2f}","clip": int(is_clipped),}
            if epoch >= warmup_epochs:
                post["rawpga"] = f"{raw_pga:.4f}"
            pbar.set_postfix(post)

            optimizer.step()
            if epoch >= warmup_epochs:
                optimizer_pga.step()

            # 窗口统计
            win_cls += cls_loss.item()
            win_pga += loss_pga.item()
            win_grad += float(grad_norm)
            win_clip += int(grad_norm > max_grad_norm)
            win_step += 1
            step += 1

            if step % log_step == 0 and step:
                best_th, best_acc = evaluate(model=model, head=head, pairs_file=pairs_file, root=lfw_test_root, device=device)
                avg_cls = win_cls / win_step
                avg_pga = win_pga / win_step
                clip_ratio = win_clip / win_step
                avg_grad_norm = win_grad / win_step

                writer.add_scalars("Loss/pga_raw", {
                    "rawk": win_rawk / win_step,
                    "rawz": win_rawz / win_step, 
                    "rawpga": win_rawpga / win_step
                }, step // log_step - 1)

                writer.add_scalars("Loss/train", {
                    "cls": avg_cls,
                    "pga": avg_pga,
                    "grad": avg_grad_norm,
                    "cratio": clip_ratio
                }, step // log_step - 1)
                writer.add_scalar("Acc/val", best_acc, step // log_step - 1)
                lr = optimizer.param_groups[0]["lr"]
                lr_pga = optimizer_pga.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[2]["lr"]
                print(
                    f"epoch {epoch:03d} | cls={avg_cls:.4f} | pga={avg_pga:.4f} | val_acc={best_acc * 100:.2f}% | best_th={best_th:.4f} | lr={lr:.8f} | lr_pga={lr_pga:.8f} |"
                    f"lr_head={lr_head:.8f} | lambda_K={lambda_align_K:.4f} | lambda_Z={lambda_align_Z:.4f} | clip_ratio={clip_ratio:.3f} | grad={avg_grad_norm:.2f}"
                )
                ckpt = {
                    "epoch": epoch,
                    "model_backbone": model.state_dict(),
                    "head": head.state_dict(),
                    "pga": pga.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer_pga": optimizer_pga.state_dict(),
                    "scheduler_pga": scheduler_pga.state_dict()
                }
                torch.save(ckpt, f"./checkpoints/pga_casia_{epoch}.pth")

                win_cls = win_pga = win_grad = 0.0
                win_clip = win_step = 0
                win_rawk = win_rawz = win_rawpga = 0
        scheduler.step()
        if epoch >= warmup_epochs:
            scheduler_pga.step()
        # best_th, best_acc = evaluate(
        #     model=model, 
        #     head=head,
        #     pairs_file=pairs_file,
        #     root=lfw_test_root,
        #     device=device
        # )
        # avg_cls = total_cls_loss / len(loader)
        # avg_pga = total_pga_loss / len(loader)
        # clip_ratio = clipped_cnt / len(loader)
        # avg_grad_norm = grad_norm_sum / len(loader)
        # writer.add_scalars("Loss/train", {
        #     "cls": avg_cls,
        #     "pga": avg_pga,
        #     "grad": avg_grad_norm,
        #     "cratio": clip_ratio
        # }, epoch)
        # writer.add_scalar("Acc/val", best_acc, epoch)
        # lr = optimizer.param_groups[0]["lr"]
        # lr_pga = optimizer_pga.param_groups[0]["lr"]
        # lr_head = optimizer.param_groups[2]["lr"]
        # print(
        #     f"epoch {epoch:03d} | cls={avg_cls:.4f} | pga={avg_pga:.4f} | val_acc={best_acc * 100:.2f}% | best_th={best_th:.4f} | lr={lr:.8f} | lr_pga={lr_pga:.8f} |"
        #     f"lr_head={lr_head:.4f} | lambda_align_K={lambda_align_K:.4f} | lambda_align_Z={lambda_align_Z:.4f} | clip_ratio={clip_ratio:.3f} | avg_gnorm={avg_grad_norm:.2f}"
        # )
        # ckpt = {
        #     "epoch": epoch,
        #     "model_backbone": model.state_dict(),
        #     "head": head.state_dict(),
        #     "pga": pga.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        #     "scheduler": scheduler.state_dict(),
        #     "optimizer_pga": optimizer_pga.state_dict(),
        #     "scheduler_pga": scheduler_pga.state_dict()
        # }
        # torch.save(ckpt, f"./checkpoints/pga_casia_{epoch}.pth")
        # scheduler.step()
        # if epoch >= warmup_epochs:
        #     scheduler_pga.step()
    writer.close()