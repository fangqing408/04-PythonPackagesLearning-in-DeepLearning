import os
import time
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
import numpy as np
import cv2
import random
import psutil
import GPUtil
from tqdm import tqdm
   
# 自定义 Dataset
class ImageDataset(Dataset):
    # 新增 files=None，可直接使用传入的子集文件列表
    def __init__(self, img_dir, method, files=None):
        self.img_dir = img_dir
        self.method = method
        if files is not None:
            self.files = files
        else:
            self.files = [
                os.path.join(root, f)
                for root, _, files in os.walk(img_dir)
                for f in files if f.lower().endswith((".jpg", ".png"))
            ]

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if self.method == "torchvision":
            img = read_image(path)
            if img.shape[0] == 1: # 灰度
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4: # RGBA
                img = img[:3, :, :]
            return img
        elif self.method == "PIL":
            img = np.array(Image.open(path).convert("RGB"))
        elif self.method == "OpenCV":
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img)

def warmup_cache(files):
    print("Warming up OS page cache (preloading images into memory)...")
    t0 = time.perf_counter()
    for path in tqdm(files, desc="Warm-up"):
        _ = read_image(path)
    t1 = time.perf_counter()
    warmup_time = t1 - t0
    print(f"Warm-up done in {warmup_time:.2f}s. All images have been cached into OS memory.\n")
    return warmup_time

def get_file_subset(img_dir, target_gb):
    files = []
    total_bytes = 0
    for root, _, fs in os.walk(img_dir):
        for f in fs:
            if not f.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            files.append(path)
            total_bytes += size
            if total_bytes >= target_gb * (1024 ** 3):
                break
        if total_bytes >= target_gb * (1024 ** 3):
            break
    print(f"Selected {len(files)} images (~{total_bytes/1024**3:.2f} GB).")
    random.shuffle(files)
    return files

def log_system_status():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = mem.used / (1024 ** 3)
    ram_percent = mem.percent
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = gpu.load * 100
        vram_used = gpu.memoryUsed / 1024
        vram_total = gpu.memoryTotal / 1024
    else:
        gpu_load = vram_used = vram_total = 0
    return cpu, ram_used, ram_percent, gpu_load, vram_used, vram_total

# 让 experiment 接收 files，并把同一子集传给 Dataset
def experiment(img_dir, files, method, num_workers, batch_size, pin_memory):
    dataset = ImageDataset(img_dir, method, files=files)  # 关键：使用同一份子集
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(f"method: {method}, num_workers: {num_workers}, batch_size: {batch_size}, pin_memory: {pin_memory}")
    t0 = time.perf_counter()
    total_images = 0
    for batch in loader:
        total_images += batch.shape[0]
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_batch_time = total_time / len(loader)
    throughput = total_images / total_time

    # 每次实验记录系统状态
    cpu, ram_used, ram_percent, gpu_load, vram_used, vram_total = log_system_status()

    print(f"Total: {total_time:.2f}s | Avg batch: {avg_batch_time:.3f}s | "
          f"Throughput: {throughput:.1f} img/s | "
          f"CPU: {cpu:.1f}% | RAM: {ram_used:.2f}GB ({ram_percent:.1f}%) | "
          f"GPU: {gpu_load:.1f}% | VRAM: {vram_used:.2f}/{vram_total:.2f} GB")

    return total_time, avg_batch_time, throughput, cpu, ram_used, ram_percent, gpu_load, vram_used, vram_total

def main():
    img_dir = "./CASIA-WebFace-Expanded"
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    # 纵轴代表消耗的时间，横轴代表 batch_size，每张图里面 30 条折线图，不同区域的标注出来代表不同的 method，每种 method 十条折线图，实线代表 pin_memory=True，虚线代表 False
    # 颜色不同代表不同的 num_workers，一共三张图，代表实验数据集大小为 4GB（远小于 page cache 大小），基本没有 I/O，20GB（与 page cache 相近），部分 I/O
    # 40GB（远大于 page cache），I/O 和直接读取很均衡，也基本不会出现 swap 现象，能较好的模拟真实的情况，但是不一定能跑完，无法跑完的话可以综合前两个大小的数据集综合得出结论

    # 一个 batch 算完之后，这个 batch 就被从显存里面清除了，下一个 batch 会覆盖同样的显存空间，单个样本读取之后会放到显存的 page cache 里面，拼接的 batch 临时创建用完即丢弃，不会缓存
    # 下次用的时候还需要重新拼接，当时样本数据可能还存在 page cache 里面，这也就是 num_workers 比较重要的原因，显存里面主要存储的就是网络参数和一个 batch 的数据还有前向传播的中间结果等

    # 训练过程中发现 OpenCV 及其的不稳定，在相同的参数下，其相邻的两次读取甚至能相差数十倍，查询资料后发现，OpenCV 使用 libjepg/libpng 等底层的 C 库进行解码，这些库本身不是线程安全的
    # 尤其是 libjepg 的旧版本，如果多线程同时读取图片，可能出现内存崩溃和数据混乱，尤其在 Windows 上表现最明显，OPenCV 的 cv2.imread 默认是 C++ 层直接读取文件，不经过 python 的 GIL
    # 所以并行安全依赖底层库，OpenCV 多线程读小规模图片通常稳定，但在高并发（> 8 线程）或者大批量读取时偶尔出错，所以深度学习场景下很少使用 OpenCV 读取文件
    # 一次出错可能导致后续一段时间都会变慢，慢慢的恢复，极少数情况下可能导致读取文件错误，但几乎不会影响训练的正确性

    methods = ["torchvision", "PIL", "OpenCV"]
    num_workers_list = [2, 4, 8] # 20GB 的测试删掉了 num_workers=12 的情况
    batch_sizes = [256] # 20GB 的测试删掉了 batch_size=64、128、384、512 的情况
    pin_memory_list = [True] # 20GB 的测试删掉了 pin_memory=False 的情况
    subset_sizes = [20]  # test dataset subsets (GB) 4 GB 已经过测试，现在测试 20 GB 的即可，40 GB 的直接丢弃

    # open() 打开一个文件并返回文件对象，参数为 path、mode 等
    save_path = os.path.join(save_dir, "benchmark_results.csv")
    if not os.path.exists(save_path):
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset_size(GB)", "method", "num_workers", "batch_size", "pin_memory",
                "warmup_time(s)",
                "total_time(s)", "avg_batch_time(s)", "throughput(img/s)",
                "CPU(%)", "RAM_used(GB)", "RAM(%)", "GPU(%)", "VRAM_used(GB)", "VRAM_total(GB)"
            ])

    for subset_gb in subset_sizes:
        print(f"\n================ Testing subset: {subset_gb} GB ================")
        files = get_file_subset(img_dir, subset_gb)
        # warmup_time = warmup_cache(files)

        results = []
        total_exps = len(methods) * len(num_workers_list) * len(batch_sizes) * len(pin_memory_list)
        pbar = tqdm(total=total_exps, desc=f"{subset_gb}GB experiments", ncols=100)

        for method in methods:
            for num_workers in num_workers_list:
                for batch_size in batch_sizes:
                    for pin_memory in pin_memory_list:
                        total_time, avg_batch_time, throughput, cpu, ram_used, ram_percent, gpu_load, vram_used, vram_total = experiment(
                            img_dir, files, method, num_workers, batch_size, pin_memory  # 关键：把同一 files 传进去
                        )
                        # 实时写入 CSV
                        with open(save_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                subset_gb, method, num_workers, batch_size, pin_memory,
                                warmup_time,
                                total_time, avg_batch_time, throughput,
                                cpu, ram_used, ram_percent, gpu_load, vram_used, vram_total
                            ])
                        results.append([
                            method, num_workers, batch_size, pin_memory,
                            total_time, avg_batch_time, throughput
                        ])
                        pbar.update(1)

        pbar.close()

    print(f"\nAll experiments finished! Results saved at: {save_path}")


if __name__ == "__main__":
    # 最安全设置：防止 Windows 多进程递归死锁
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
