import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# ==================== 基础函数区 ======================
# ======================================================
# 计算样本两两之间的余弦相似度矩阵
def pairwise_cosine(X, eps=1e-8):
    X = F.normalize(input=X, dim=-1) # [B, D]
    sim = X @ X.t()
    return sim.clamp(min=-1.0 + eps, max=1.0 - eps) # [B, B]

# 选取 top-k 相似的邻居，sim 是相似度矩阵，base_mask 1 代表允许成为侯选边，0 代表不允许成为侯选边
def _knn_mask(sim, base_mask, device, topk=8):
    # B 也就是 batch 的大小
    B = sim.size(dim=0) 
    # 去掉自环，不许把自己选成自己的邻居，对角线上减去了一个极大的数字，后续 topk 不会选到他
    # 把不允许成为侯选边的设置为极小值，不会被 topk 取到
    sim_no_diag = sim - torch.eye(n=B, device=device) * 1e9
    masked = sim_no_diag * base_mask - (1.0 - base_mask) * 1e9
    topk = min(topk, max(B - 1, 1))
    # 每个维度从大到小取出前 k 个数，返回这 k 个数和他们的索引，一般用二元组承接，可以 .indices 只要索引
    idx = torch.topk(
        masked, 
        k=topk, 
        dim=1
    ).indices
    m = torch.zeros_like(input=sim)
    # dim, idx, value，在 dim 维度上，将  idx 指定的位置置为 value
    m.scatter_(1, idx, 1.0)
    # 因为选取前 k 个可能得到的矩阵的单向边，这里将其变为对称矩阵
    m = torch.maximum(m, m.t())
    return m

# 下面的写法经典又巧妙，.unsqueeze 在指定的维度上插入一个新维度，下面的写法利用广播得到了同类样本矩阵
# 构建类内类间图邻接，暂时把 topk 置为 8，batch_size 置为 128
def build_intra_inter(sim, labels, device, topk=8):
    same = (labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)).float()
    diff = 1.0 - same
    M_intra = _knn_mask(
        sim=sim, 
        base_mask=same, 
        device=device, 
        topk=topk
    ) # 类内图前 topk 同类邻居
    M_inter = _knn_mask(
        sim=sim, 
        base_mask=diff, 
        device=device, 
        topk=topk
    ) # 类间图前 topk 异类邻居
    # 只保留正相似，避免噪声边，全都保留正的数值，越大越相似
    A_intra = (sim * M_intra).clamp(min=0.0)
    A_inter = (sim * M_inter).clamp(min=0.0)
    return A_intra, A_inter

# 做图的对称归一化，节点的度越大，信息扩散的时候就会被平均的越多，所以归一化，来平衡不同节点的影响
def normalize_sym(A, eps=1e-8):
    # D^{-1/2} A D^{-1/2}
    d = A.sum(dim=1).clamp_min(eps)
    Dinv = torch.diag(input=torch.pow(d, -0.5))
    return Dinv @ A @ Dinv

def diffusion_kernel(A_norm, t_diff=1):
    # 和邻居信息进行融合，此时每个节点的影响范围是一跳的邻居，每循环一次就在图上多传播一层
    # 这个是先计算相似度矩阵的连乘，最后乘在节点矩阵上面进行特征的融合
    K = A_norm
    for _ in range(max(t_diff - 1, 0)):
        K = K @ A_norm
    return K

# 我们希望我们的相似性矩阵逼近一个理想矩阵，也就是下面构造的 Gidea，其实最理想的就是同类为 1，异类为 0，这里加上了一点点的平滑
def build_idea(labels, sigma_in=0.99, sigma_out=0.00):
    # 生成固定的理想图（仅用于最后一层对齐）
    same = (labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)).float()
    diff = 1.0 - same
    A = same * sigma_in + diff * sigma_out
    return A

# ======================================================
# =====================GCN 模块=========================
# ======================================================
class GAM(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=dim, 
            out_features=dim, 
            bias=False
        )
        self.fc2 = nn.Linear(
            in_features=dim, 
            out_features=dim, 
            bias=False
        )
        self.bn  = nn.BatchNorm1d(num_features=dim)

    def forward(self, A_norm, X):
        # X: [B,D], A_norm: [B,B]
        Z = A_norm @ self.fc1(X)
        Z = F.relu(input=self.bn(Z))
        Z = A_norm @ self.fc2(Z)
        return Z + X
    
# ======================================================
# =====================PGA 主结构=======================
# ======================================================
class PGAHead(nn.Module):
    def __init__(self, num_layers, device, topk=8, t_diff=1, use_ema=True, ema_m=0.9):
        super().__init__()
        self.gams = nn.ModuleList([GAM(512) for _ in range(num_layers)])
        self.topk = topk
        self.t_diff = t_diff
        self.ema_m = ema_m
        self.device = device
        self.num_layers = num_layers
        # α, β 可为标量或每层独立的 list/张量（默认类内↑类间↓）
        self.alpha_sched = torch.linspace(
            start=1.00, 
            end=1.20, 
            steps=num_layers
        ).tolist()
        self.beta_sched  = torch.linspace(
            start=0.00, 
            end=0.00, 
            steps=num_layers
        ).tolist()
        # EMA 缓冲区（更稳的目标结构）
        # 我们不直接拿当前的 batch 算出来的 K 对齐，而是对他做一个平滑
        # 是否启用 EMA 平滑，方便打开和关闭
        self.use_ema = use_ema
        self.register_buffer("initialized", torch.tensor(0))
        self._ema_K = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1), requires_grad=False)
            # 存放每层的平均图结构，保证张量随着设备移动，占位符 (1, 1) 代表还没拿到真实的 K
            # requires_grad 表示这些不是可学习的参数只是简单的缓存
            for _ in range(num_layers)
        ])
        # 投影层，将每层 Z 投影到统一 768 维空间以进行语义对齐
        self.proj = nn.Linear(
            in_features=512, 
            out_features=768, 
            bias=False
        )

    def set_alpha_beta(self, alpha, beta):
        self.alpha_sched = alpha
        self.beta_sched = beta

    def _get_alpha_beta(self, layer_idx):
        def pick(val):
            if isinstance(val, (list, tuple)):
                return torch.as_tensor(val[layer_idx], device=self.device, dtype=torch.float32)
            if torch.is_tensor(val) and val.ndim > 0:
                return val[layer_idx].to(device=self.device, dtype=torch.float32)
            return torch.tensor(float(val), device=self.device)
        a = torch.clamp(pick(self.alpha_sched), min=0.0)
        b = torch.clamp(pick(self.beta_sched), min=0.0)
        return a, b
    
    def _graph(self, X, labels, a, b):
        S = pairwise_cosine(X)
        A_intra, A_inter = build_intra_inter(
            sim=S, 
            labels=labels, 
            device=self.device, 
            topk=self.topk
        )
        # 加一点自环，避免度为 0
        A = a * A_intra + b * A_inter + 1e-6 * torch.eye(n=S.size(0), device=self.device)
        A_norm = normalize_sym(A)
        K = diffusion_kernel(A_norm=A_norm, t_diff=self.t_diff)
        return A_norm, K
    
    def _init_ema_if_needed(self, K_list):
        if int(self.initialized.item()) == 1: return 
        # 开始跑到 时候创建副本，后续不再进行更改
        with torch.no_grad():
            for i, K in enumerate(K_list):
                # 遍历每层的真实图 K 从计算图里面分离并复制一份不会参与反向传播，存储值副本
                self._ema_K[i] = nn.Parameter(K.detach().clone(), requires_grad=False)
            self.initialized.fill_(1)

    def forward(self, feats_final, labels, lambda_align_K=64, lambda_align_Z=16, lambda_idea=1.0, sigma_in=0.99, sigma_out=0.00, stopgrad=True):
        A_list, K_list, Z_list = [], [], []
        for i, (X, gam) in enumerate(zip(feats_final, self.gams)):
            a_i, b_i = self._get_alpha_beta(layer_idx=i)
            Ai, Ki = self._graph(
                X=X, 
                labels=labels, 
                a=a_i, 
                b=b_i
            )
            Zi = gam(Ai, X)
            A_list.append(Ai); K_list.append(Ki); Z_list.append(Zi) 
        if self.use_ema:
            self._init_ema_if_needed(K_list)
            with torch.no_grad():
                for i in range(self.num_layers):
                    self._ema_K[i].data.mul_(self.ema_m).add_(K_list[i].detach() * (1.0 - self.ema_m))
        # 逐层定向对齐：|| K_{i-1} - sg[K_i] ||^2 和 || Z_{i-1} - sg[Z_i] ||^2
        loss_align_K = torch.zeros((), device=self.device)
        loss_align_Z = torch.zeros((), device=self.device)
        for i in range(1, self.num_layers):
            Ki_prev = K_list[i - 1]
            Ki_tgt  = self._ema_K[i].detach() if self.use_ema else (K_list[i].detach() if stopgrad else K_list[i])
            loss_align_K = loss_align_K + (Ki_prev - Ki_tgt).pow(2).mean()
            Zi_prev = F.normalize(input=self.proj(Z_list[i - 1]), dim=-1)
            Zi_curr = F.normalize(input=self.proj(Z_list[i]), dim=-1)
            loss_align_Z = loss_align_Z + (Zi_prev - Zi_curr.detach()).pow(2).mean()
        K_out  = K_list[-1]
        K_idea = build_idea(
            labels=labels, 
            sigma_in=sigma_in, 
            sigma_out=sigma_out
        ).to(self.device)
        loss_idea = (K_out - K_idea).pow(2).mean()
        losses = {
            "loss_align_K": loss_align_K,
            "loss_align_Z": loss_align_Z,
            "loss_idea":  loss_idea,
            "loss_pga":   lambda_align_K * loss_align_K + 
                          lambda_align_Z * loss_align_Z +
                          lambda_idea * loss_idea
        }
        return losses