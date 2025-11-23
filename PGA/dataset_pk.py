import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from PIL import Image
import os, math
import random
import torchvision.transforms as T

class PKDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        classes = sorted(os.listdir(root))
        self.num_classes = len(classes)
        for label, cls in enumerate(classes):
            path = os.path.join(root, cls)
            self.class_to_idx[cls] = label
            for fname in os.listdir(path):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(path, fname)
                    self.samples.append((img_path, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label, idx

class PKSampler(Sampler):
    def __init__(self, data, P=16, K=16, shuffle=True):
        self.data = data
        self.P = P
        self.K = K
        self.shuffle = shuffle
        self.lab2idx = {}
        # 建立 label -> indices 映射
        for idx, (_, label) in enumerate(data.samples):
            self.lab2idx.setdefault(label, []).append(idx)
        self.labels_unique = list(self.lab2idx.keys())
        # self.lab2idx
        # {
        #     0: [0, 5, 12, 27, ...],
        #     1: [1, 9, 18, ...],
        #     2: [2, 6, 8, ...],
        #     ...
        # }
        # self.labels_unique [0, 1, 2, ...], 随机打乱标签抽取 P 个，每类再取 K 个样本
    def __iter__(self):
        # 拷贝，避免修改原列表，打乱的话，就在类级别打乱
        classes = self.labels_unique.copy()
        if self.shuffle:
            random.shuffle(classes)
        batch = []
        # 步长为 P，每次切出 P 个类
        for i in range(0, len(classes), self.P):
            cls_batch = classes[i:i + self.P]
            # 不足 P 个，从所有类里面随机补到 P 个
            if len(cls_batch) < self.P:
                cls_batch += random.sample(self.labels_unique, self.P - len(cls_batch))
            for c in cls_batch:
                # 取出这类的所有样本，足够长，无放回随机抽 K 个，不足的话就有放回抽样
                idx_pool = self.lab2idx[c]
                if len(idx_pool) >= self.K:
                    chosen = random.sample(idx_pool, self.K)
                else:
                    chosen = random.choices(idx_pool, k=self.K)
                # 凑好的 batch 返回，再次将 batch 置为空
                batch.extend(chosen)
            random.shuffle(batch)
            yield batch
            batch = []

    def __len__(self):
        # 不是样本数，而是批次数的估计值，最后一个可能因为补类而略有误差，不过对 DataLoader 来说够用了
        return math.ceil(len(self.labels_unique) / self.P)
    
# ======================================================
# =============== 验证数据集的创建情况 ==================
# ======================================================
if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = PKDataset("./CASIA-WebFace", transform=transform)
    sampler = PKSampler(dataset, P=8, K=8)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    loader_iter = iter(loader)
    for i in range(4):
        imgs, labels = next(loader_iter)
        print(f"batch {i}: imgs.shape={imgs.shape}, unique_classes={len(torch.unique(labels))}")
        print(labels)