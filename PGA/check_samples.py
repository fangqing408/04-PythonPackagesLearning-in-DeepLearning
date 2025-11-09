# post_filter_bad.py  —— 只复核 bad_images.csv，并判断“图坏还是变换导致”
# 正常样本不打印；确认异常才打印少量信息 + 生成 bad_images_clean.csv + 可选隔离拷贝

import os, csv, shutil, random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
# 顶部 imports 里补一行：
import torchvision.transforms.functional as F

import torch
import torchvision.transforms as T

# ===== 配置 =====
DATA_ROOT = "./CASIA-WebFace"
IN_CSV    = "./bad_images.csv"          # 已经跑出来的初始清单（可能含误报）
OUT_CSV   = "./bad_images_clean.csv"    # 复核后的“真坏图”清单（含原因/阶段）
QUAR_DIR  = "./quarantine"            # 把真坏图拷到这里（保持目录结构）；不需要可设为 None

# 与训练一致的形状
CROP_SIZE = (128, 128)

# 基础算子（诊断会分阶段使用）
_resize    = T.Resize((144, 144))
_center    = T.CenterCrop(CROP_SIZE)
_totensor  = T.ToTensor()
_norm      = T.Normalize(mean=[0.5]*3, std=[0.5]*3)
_rcrop     = T.RandomCrop(CROP_SIZE)
_rflip     = T.RandomHorizontalFlip(p=0.5)

def open_rgb(path: str):
    with Image.open(path) as im:
        im.verify()  # 头部校验
    return Image.open(path).convert("RGB")

def is_nonfinite(x: torch.Tensor):
    return bool((~torch.isfinite(x)).any())

def first_coords(x: torch.Tensor, n=8):
    idx = torch.nonzero(~torch.isfinite(x), as_tuple=False)
    pairs = []
    for i in range(min(n, idx.size(0))):
        c,y,xx = idx[i].tolist()
        pairs.append(((c,y,xx), x[c,y,xx].item()))
    return pairs, int(idx.size(0))

def diagnose_image(path: str):
    """
    分阶段、确定性复核（不走“随机”运气）：
    A) Resize→CenterCrop→ToTensor：若坏 => decode/totensor
    B) 上面再 Normalize：若坏 => normalize
    C) Resize 后对所有 17×17 个 (128×128) 裁剪位置做穷举，
       同时考虑不翻转/水平翻转两种情况（总 578 块）；
       只要有一块 Normalize 后非有限 => randomcrop/flip
    返回：
      (is_bad, reason, stage, nf_count, samples)
    """
    try:
        im = open_rgb(path)                              # PIL
        im_resized = _resize(im)                         # 144x144 (PIL)

        # ---------- 阶段 A：ToTensor 前就坏？ ----------
        im_center = _center(im_resized)                  # PIL 128x128
        x1 = _totensor(im_center).float()                # [C,H,W] ∈ [0,1]
        if is_nonfinite(x1):
            samples, cnt = first_coords(x1, n=8)
            return True, "nonfinite_before_normalize", "decode/totensor", cnt, samples

        # ---------- 阶段 B：Normalize 后才坏？ ----------
        x2 = _norm(x1.clone())
        if is_nonfinite(x2):
            samples, cnt = first_coords(x2, n=8)
            return True, "nonfinite_after_normalize", "normalize", cnt, samples

        # ---------- 阶段 C：穷举所有裁剪位置 + 是否翻转 ----------
        W, H = im_resized.size  # 应该是 (144, 144)
        crop_w, crop_h = CROP_SIZE[1], CROP_SIZE[0]  # 128, 128
        max_left  = W - crop_w                       # 16
        max_top   = H - crop_h                       # 16

        for flip in (False, True):
            src = F.hflip(im_resized) if flip else im_resized  # PIL
            # 穷举所有 top/left（步长为 1）
            for top in range(max_top + 1):
                for left in range(max_left + 1):
                    patch = F.crop(src, top=top, left=left, height=crop_h, width=crop_w)  # PIL
                    x = _norm(_totensor(patch).float())
                    if is_nonfinite(x):
                        samples, cnt = first_coords(x, n=8)
                        tag = f"nonfinite_on_gridcrop(top={top},left={left},flip={int(flip)})"
                        return True, tag, "randomcrop/flip", cnt, samples

        # 全部通过即正常
        return False, "", "", 0, []

    except Exception as e:
        return True, f"open_error:{type(e).__name__}:{e}", "decode", 0, []


def ensure_dir_for(dst_path: str):
    d = os.path.dirname(dst_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    if not os.path.exists(IN_CSV):
        print("找不到 bad_images.csv，等扫描脚本跑完再复核。"); return

    with open(IN_CSV, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r, None)
        rows = [(p, reason) for p, reason in r]

    print(f"待复核：{len(rows)} 张（仅复核 CSV 里的，不重扫全库）")

    kept = []  # (path, reason, stage, nf_count)
    for p, _ in rows:
        if not os.path.exists(p):
            kept.append((p, "missing_file", "decode", 0))
            continue

        bad, reason, stage, cnt, samples = diagnose_image(p)
        if bad:
            kept.append((p, reason, stage, cnt))
            # 仅在确认坏图时打印少量信息
            print(f"❌ 确认坏图: {p}  stage={stage}  count={cnt}  samples={samples[:3]}")
            # 可选：隔离拷贝
            if QUAR_DIR:
                rel = os.path.relpath(p, DATA_ROOT)
                dst = os.path.join(QUAR_DIR, rel)
                ensure_dir_for(dst)
                try:
                    shutil.copy2(p, dst)
                except Exception as e:
                    print(f"  搬运失败: {p} -> {dst} ({e})")
        # 正常的不打印

    # 写出干净版清单（含阶段/数量）
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason", "stage", "nonfinite_count"])
        w.writerows(kept)

    # 小结
    confirmed = sum(1 for _, r, _, _ in kept if r and not r.startswith("missing_file"))
    print(f"复核完成：确认坏图 {confirmed} 张；结果写入 {OUT_CSV}"
          + (f"，坏图副本在 {QUAR_DIR}/" if QUAR_DIR else ""))

if __name__ == "__main__":
    main()
