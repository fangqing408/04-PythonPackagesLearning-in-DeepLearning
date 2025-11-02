import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from PIL import Image
import os
import random

class OverlapDataset(Dataset):
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
        img = Image.open(img_path).convert("L")
        if self.transform: img = self.transform(img)
        return img, label

class OverlapSampler(Sampler):
    def __init__(self, data, batch_size, overlap_ratio=0.4, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.overlap_ratio = overlap_ratio
        self.step = int(batch_size * (1 - overlap_ratio))
        self.shuffle = shuffle
        self.indices = [
            list(range(i, i + batch_size))
            for i in range(0, len(data) - batch_size + 1, self.step)
        ]
    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle: random.shuffle(indices)
        for i in range(0, len(self.data) - self.batch_size + 1, self.step):
            batch_idxs = indices[i:i + self.batch_size]
            yield batch_idxs
    def __len__(self):
        return (len(self.data) - self.batch_size) // self.step + 1

# ======================================================
# ===============验证数据集的创建情况====================
# ======================================================
if __name__ == "__main__":
    images = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 10, (100, ))

    dataset = OverlapDataset(images, labels)
    sampler = OverlapSampler(dataset, batch_size=8, overlap_ratio=0.4)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)

    # DataLoader 的底层逻辑的先看你有没有提供自己的 Sampler，否则的话自己创建随机的采样，当循环遍历 loader 的时候，这个迭代器会不断的调用 indices=next(iter(sampler))
    # 和 Dataset 类似，sampler 必须实现两个函数 __iter__ 和 __len__，__iter__ 返回的一堆索引，也就是 DataLoader 加载数据的索引

    for step, (img, label) in enumerate(loader):
        print(f"step {step}: img={img.shape}, label={label.tolist()}")