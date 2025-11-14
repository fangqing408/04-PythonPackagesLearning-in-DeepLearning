import math, random
from torch.utils.data import Sampler

class PKMixSampler(Sampler):
    def __init__(self, data, P=16, K=16, random_size=256, shuffle=True, avoid_overlap=True):
        self.data = data
        self.P = int(P)
        self.K = int(K)
        self.random_size = int(random_size)
        self.shuffle = shuffle
        self.avoid_overlap = avoid_overlap

        # 建立 label -> indices 映射
        self.lab2idx = {}
        for idx, (_, label) in enumerate(data.samples):
            self.lab2idx.setdefault(int(label), []).append(idx)
        self.labels_unique = list(self.lab2idx.keys())

        # 全局样本数量
        self.N = len(self.data)

    def __iter__(self):
        classes = self.labels_unique.copy()
        if self.shuffle:
            random.shuffle(classes)

        batch = []
        # 步长为 P，每次切出 P 个类
        for i in range(0, len(classes), self.P):
            cls_batch = classes[i:i + self.P]
            # 不足 P 个，从所有类随机补到 P 个
            if len(cls_batch) < self.P:
                cls_batch += random.sample(self.labels_unique, self.P - len(cls_batch))

            # 1) 先取 PK 子块：每类取 K
            pk_indices = []
            for c in cls_batch:
                idx_pool = self.lab2idx[c]
                if len(idx_pool) >= self.K:
                    chosen = random.sample(idx_pool, self.K)     # 无放回
                else:
                    chosen = random.choices(idx_pool, k=self.K) # 有放回补齐
                pk_indices.extend(chosen)

            # 2) 再取随机子块
            rand_indices = []
            if self.random_size > 0:
                rand_indices = random.sample(range(self.N), min(self.random_size, self.N))
                # 避免与 PK 子块重复（开销很小）
                if self.avoid_overlap and len(pk_indices) > 0:
                    pk_set = set(pk_indices)
                    for j in range(len(rand_indices)):
                        if rand_indices[j] in pk_set:
                            # 找一个不在 pk_set 的替代，最多尝试 10 次
                            tries = 0
                            while tries < 10:
                                cand = random.randrange(self.N)
                                if cand not in pk_set:
                                    rand_indices[j] = cand
                                    break
                                tries += 1

            # 3) 合并并打乱（你特别强调的点）
            batch = pk_indices + rand_indices
            if self.shuffle:
                random.shuffle(batch)

            yield batch
            batch = []

    def __len__(self):
        # 与你的 PKSampler 一致：按类的分块估计批次数
        return math.ceil(len(self.labels_unique) / self.P)
