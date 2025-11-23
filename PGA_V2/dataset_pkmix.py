from torch.utils.data import Dataset, Sampler
from PIL import Image
import os, random

class PKMixDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root))
        self.num_classes = len(classes)

        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            self.class_to_idx[cls] = label
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_dir, fname)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # idx 作为 sample_id，给 PGA 的 EMA 使用
        return img, label, idx

class PKMixSampler(Sampler):

    def __init__(self, data, batch_main_size=448, P=8, K=8, shuffle=True, avoid_overlap=True):
        self.data = data
        self.batch_main_size = int(batch_main_size)
        self.P = int(P)
        self.K = int(K)
        self.shuffle = shuffle
        self.avoid_overlap = avoid_overlap

        self.N = len(self.data)
        self.all_indices = list(range(self.N))

        self.batch_pk_size = self.P * self.K
        self.batch_total_size = self.batch_main_size + self.batch_pk_size

        # label -> indices
        self.lab2idx = {}
        for idx, (_, label) in enumerate(data.samples):
            self.lab2idx.setdefault(int(label), []).append(idx)
        self.labels_unique = list(self.lab2idx.keys())

    def __iter__(self):
        indices = self.all_indices.copy()
        if self.shuffle:
            random.shuffle(indices)

        # 按 total_size=512 一块一块地走
        for start in range(0, self.N, self.batch_total_size):
            base_chunk = indices[start:start + self.batch_total_size]
            # 为了保证 batch 大小严格一致，尾巴不够 512 的直接丢弃（和 drop_last=True 一样）
            if len(base_chunk) < self.batch_total_size:
                continue

            # 1) 在这 512 里面随机保留 448 个作为主随机部分，其余 64 就“丢掉”
            #    （这 448 就是你之前的 batch_main_size）
            batch_random = random.sample(base_chunk, self.batch_main_size)
            random_set = set(batch_random)

            # 2) PK 附加部分：P * K
            pk_indices = []
            if len(self.labels_unique) > 0 and self.P > 0 and self.K > 0:
                # 随机挑 P 个类
                cls_batch = random.sample(
                    self.labels_unique,
                    k=min(self.P, len(self.labels_unique))
                )
                for c in cls_batch:
                    idx_pool = self.lab2idx[c]  # 该类所有样本 index

                    if self.avoid_overlap:
                        # 优先挑“不在随机部分”的
                        pool_wo_random = [i for i in idx_pool if i not in random_set]
                    else:
                        pool_wo_random = idx_pool

                    if len(pool_wo_random) >= self.K:
                        chosen = random.sample(pool_wo_random, self.K)
                    elif len(idx_pool) >= self.K:
                        # 不够就退一步，在整类里随机（可能和 random 部分重叠）
                        chosen = random.sample(idx_pool, self.K)
                    else:
                        # 类本身样本数太少，只能有放回采样
                        chosen = random.choices(idx_pool, k=self.K)

                    pk_indices.extend(chosen)

            # 3) 合并：448 random + 64 PK = 512
            batch = batch_random + pk_indices

            # 可选：如果你希望 batch 内样本顺序再打散，就保留这句
            random.shuffle(batch)

            # 保险校验一下大小（正常来说一定是 512）
            if len(batch) == self.batch_total_size:
                yield batch

    def __len__(self):
        # 用 512 步长估计 iter 数，和基线 (batch=512, drop_last=True) 对齐
        return self.N // self.batch_total_size
