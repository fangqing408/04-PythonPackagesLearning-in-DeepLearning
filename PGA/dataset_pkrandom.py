from torch.utils.data import Dataset, Sampler
from PIL import Image
import os, math, random

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
    """
    随机为主 + PK 附加：
      - 每个 iter:
          先取 batch_main_size 个随机样本作为分类主 batch
          再为 PGA 额外采 P 个类，每类 K 个样本 => P*K 个 PK 样本
          最终返回的 batch = [random 部分] + [PK 部分]
      - 注意：这里不对整体 batch 做 shuffle，
        方便在训练里用前 B_main 给 CE，后 P*K 给 PGA。
    """
    def __init__(self, data, batch_main_size=448, P=8, K=8,
                 shuffle=True, avoid_overlap=True):
        self.data = data
        self.batch_main_size = int(batch_main_size)   # 给随机分类用的部分
        self.P = int(P)
        self.K = int(K)
        self.shuffle = shuffle
        self.avoid_overlap = avoid_overlap

        self.N = len(self.data)
        self.all_indices = list(range(self.N))

        # label -> indices
        self.lab2idx = {}
        for idx, (_, label) in enumerate(data.samples):
            self.lab2idx.setdefault(int(label), []).append(idx)
        self.labels_unique = list(self.lab2idx.keys())

    def __iter__(self):
        indices = self.all_indices.copy()
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, self.N, self.batch_main_size):
            # 1) 随机主 batch（分类用）
            batch_random = indices[start:start + self.batch_main_size]
            if len(batch_random) == 0:
                continue
            random_set = set(batch_random)

            # 2) PK 附加 batch（P*K，用于 PGA）
            pk_indices = []
            if len(self.labels_unique) > 0 and self.P > 0 and self.K > 0:
                cls_batch = random.sample(
                    self.labels_unique,
                    k=min(self.P, len(self.labels_unique))
                )
                for c in cls_batch:
                    idx_pool = self.lab2idx[c]      # 该类的所有样本 index

                    # 优先选择“同类 & 不在 random 部分”的样本
                    if self.avoid_overlap:
                        pool_wo_random = [i for i in idx_pool if i not in random_set]
                    else:
                        pool_wo_random = idx_pool

                    if len(pool_wo_random) >= self.K:
                        chosen = random.sample(pool_wo_random, self.K)
                    elif len(idx_pool) >= self.K:
                        # 不够就退一步，在整类里随机（可能和 random 有重合）
                        chosen = random.sample(idx_pool, self.K)
                    else:
                        # 类本身样本数太少，只能有放回采样
                        chosen = random.choices(idx_pool, k=self.K)

                    pk_indices.extend(chosen)

            # 3) 合并：前面是 random，后面是 PK，不再打乱
            
            batch = batch_random + pk_indices
            random.shuffle(batch)
            if len(batch) == self.batch_main_size + self.P * self.K:
                yield batch

    def __len__(self):
        # 按主随机部分的切片来估计 iter 数量
        return self.N // self.batch_main_size
