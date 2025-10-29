import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

# ======================================================
# ==================== 基础函数区 ======================
# ======================================================
def pairwise_cosine(x, eps=1e-8): # 计算样本两两之间的余弦相似度矩阵
    x = F.normalize(x, dim=-1) # [B, D]
    sim = x @ x.t()
    return sim.clamp(-1.0 + eps, 1.0 - eps) # [B, B] 并且加上了数值稳定性的部分

def _knn_mask(sim, base_mask, k): # 选取 top-k 相似的邻居，sim 是相似度矩阵，base_mask 1 代表允许成为侯选边，0 代表不允许成为侯选边
    B = sim.size(0) # B 也就是 Batch 的大小
    # 注意 Batch 设置的不要过小，至少应该大于 topk
    sim_no_diag = sim - torch.eye(B, device=config.device) * 1e9 # 去掉自环，不许把自己选成自己的邻居，对角线上减去了一个极大的数字，后续 topk 不会选到他
    # 保持 device 一致，防止 cpu 和 cuda 混用出现问题，因为相似度可能出现负值所以这里要置为一个极小值
    masked = sim_no_diag * base_mask - (1.0 - base_mask) * 1e9 # 把不允许成为侯选边的设置为极小值，不会被 topk 取到
    k = min(k, max(B - 1, 1)) # 这里 Batch 最好大于 topk，但是还是加上了 max 防止报错
    idx = torch.topk(masked, k=k, dim=1).indices # 每个维度从大到小取出前 k 个数，返回这 k 个数和他们的索引，一般用二元组承接，可以 .indices 只要索引
    m = torch.zeros_like(sim)
    m.scatter_(1, idx, 1.0) # dim, idx, value，在 dim 维度上，将  idx 指定的位置置为 value
    m = torch.maximum(m, m.t())  # 因为选取前 k 个可能得到的矩阵的单向边，这里将其变为对称矩阵
    return m

def build_intra_inter(sim, y, topk=8): # 构建类内类间图邻接，暂时把 topk 置为 64，Batch_size 置为 256
    # sim: [B,B], y: [B]
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float() # 构建同类集合
    diff = 1.0 - same # 构建同类集合的补集
    M_intra = _knn_mask(sim, same, topk) # 类内图前 topk 同类邻居
    M_inter = _knn_mask(sim, diff, topk) # 类间图前 topk 异类邻居
    # 只保留正相似，避免噪声边，全都保留正的数值，越大越相似
    A_intra = (sim * M_intra).clamp(min=0.0)
    A_inter = (sim * M_inter).clamp(min=0.0)
    return A_intra, A_inter

def normalize_sym(A, eps=1e-8): # 做图的对称归一化，节点的度越大，信息扩散的时候就会被平均的越多，所以归一化，来平衡不同节点的影响
    # D^{-1/2} A D^{-1/2}
    d = A.sum(dim=1).clamp_min(eps) # 每行求和
    Dinv = torch.diag(torch.pow(d, -0.5)) # 度矩阵的 -1/2 次方转为对角阵
    return Dinv @ A @ Dinv # 计算最终的结果

def diffusion_kernel(A_norm, t=1):
    # 和邻居信息进行融合，此时每个节点的影响范围是一跳的邻居，每循环一次就在图上多传播一层
    # 这个是先计算相似度矩阵的连乘，最后乘在节点矩阵上面进行特征的融合
    K = A_norm
    for _ in range(max(t - 1, 0)):
        K = K @ A_norm
    return K

# 我们希望我们的相似性矩阵逼近一个理想矩阵，也就是下面构造的 Gidea，其实最理想的就是同类为 1，异类为 0，这里加上了一点点的平滑
def build_idea(labels, sigma_in=0.99, sigma_out=0.01):
    # 生成固定的理想图（仅用于最后一层对齐）
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B,B]
    diff = 1.0 - same
    # 类内 > 1 放大相似度（更紧）
    # 类间 < 1 削弱相似度（更远）
    A = same * sigma_in + diff * sigma_out
    A = torch.maximum(A, torch.eye(A.size(0), device=config.device))
    return normalize_sym(A)
# ======================================================
# =====================GCN 模块=========================
# ======================================================
class GAM(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.bn  = nn.BatchNorm1d(dim)

    def forward(self, A_norm, X):
        # X: [B,D], A_norm: [B,B]
        Z = A_norm @ self.fc1(X)
        Z = F.relu(self.bn(Z))
        Z = A_norm @ self.fc2(Z)
        return Z + X  # 残差结构

# ======================================================
# =====================PGA 主结构=======================
# ======================================================
class PGAHead(nn.Module):
    def __init__(self, num_layers, topk=8, t_diff=1, use_ema_target=True, ema_m=0.9):
        super().__init__()
        self.gams = nn.ModuleList([GAM(512) for _ in range(num_layers)])
        self.topk = topk
        self.t_diff = t_diff

        # α, β 可为标量或每层独立的 list/张量（默认类内↑类间↓）
        self.alpha_sched = torch.linspace(1.00, 1.20, steps=num_layers).tolist()
        self.beta_sched  = torch.linspace(0.00, 0.00, steps=num_layers).tolist()

        # EMA 缓冲区（更稳的目标结构）
        # 我们不直接拿当前的 batch 算出来的 K 对齐，而是对他做一个平滑
        self.use_ema_target = use_ema_target # 是否启用 EMA 平滑，方便打开和关闭
        self.ema_m = ema_m # 平滑系数
        self.register_buffer("initialized", torch.tensor(0))  # 初始化标志，初始化代表没建立平均图
        self._ema_K = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1), requires_grad=False) # 存放每层的平均图结构，保证张量随着设备移动，占位符 (1, 1) 代表还没拿到真实的 K
            # requires_grad 表示这些不是可学习的参数只是简单的缓存
            for _ in range(num_layers)
        ]) # 初始化了 EMA 的占位符等信息，现在还没开始真正的存储平滑图

        # 投影层，将每层 Z 投影到统一 768 维空间以进行语义对齐
        self.proj = nn.Linear(512, 768, bias=False)

    def set_alpha_beta(self, alpha, beta): # 可以在外面调整调度表
        self.alpha_sched = alpha
        self.beta_sched = beta

    def _get_alpha_beta(self, layer_idx):
        def pick(val):
            if isinstance(val, (list, tuple)):
                return torch.as_tensor(val[layer_idx], device=config.device, dtype=torch.float32)
            if torch.is_tensor(val) and val.ndim > 0:
                return val[layer_idx].to(device=config.device, dtype=torch.float32)
            return torch.tensor(float(val), device=config.device)
        a = torch.clamp(pick(self.alpha_sched), min=0.0)
        b = torch.clamp(pick(self.beta_sched),  min=0.0)
        return a, b

    def _graph(self, F512, y, a, b):
        S = pairwise_cosine(F512) # 计算余弦相似度
        A_intra, A_inter = build_intra_inter(S, y, topk=self.topk) # 构建类内类间图
        # 加一点自环，避免度为 0
        A = a * A_intra + b * A_inter + 1e-6 * torch.eye(S.size(0), device=config.device)
        A_norm = normalize_sym(A) # 度归一化
        K = diffusion_kernel(A_norm, t=self.t_diff) # 传播扩散核
        return A_norm, K

    def _init_ema_if_needed(self, K_list):
        if int(self.initialized.item()) == 1: return # 开始跑到 时候创建副本，后续不再进行更改
        with torch.no_grad(): # 不记录梯度，因为只是简单的复制数据
            for i, K in enumerate(K_list):
                self._ema_K[i] = nn.Parameter(K.detach().clone(), requires_grad=False) # 遍历每层的真实图 K 从计算图里面分离并复制一份不会参与反向传播，存储值副本
            self.initialized.fill_(1)
    
    def forward(self, feats_final, labels, lambda_align_K=128, lambda_align_Z=64, lambda_idea=1.0, sigma_in=0.99, sigma_out=0.01, stopgrad=True):
        L = len(feats_final)

        A_list, K_list, Z_list = [], [], []
        for i, (F512, gam) in enumerate(zip(feats_final, self.gams)):
            a_i, b_i = self._get_alpha_beta(i) # 取出来参数
            Ai, Ki = self._graph(F512, labels, a_i, b_i) # 计算扩散核和原始核
            Zi = gam(Ai, F512) # 扩散核计算 GCN 融合邻居信息
            A_list.append(Ai); K_list.append(Ki); Z_list.append(Zi) 

        if self.use_ema_target:
            self._init_ema_if_needed(K_list)
            with torch.no_grad():
                for i in range(L):
                    self._ema_K[i].data.mul_(self.ema_m).add_(K_list[i].detach() * (1.0 - self.ema_m)) # 这个只是 detach 就行只是用了一下，没有修改 K 的值

        # 逐层定向对齐：|| K_{i-1} - sg[K_i] ||^2 和 || Z_{i-1} - sg[Z_i] ||^2
        loss_align_K = torch.zeros((), device=config.device)
        loss_align_Z = torch.zeros((), device=config.device)
        for i in range(1, L):
            Ki_prev = K_list[i - 1]
            Ki_tgt  = self._ema_K[i].detach() if self.use_ema_target else (K_list[i].detach() if stopgrad else K_list[i])
            loss_align_K = loss_align_K + (Ki_prev - Ki_tgt).pow(2).mean()

            Zi_prev = F.normalize(self.proj(Z_list[i - 1]), dim=-1)
            Zi_curr = F.normalize(self.proj(Z_list[i]), dim=-1)
            loss_align_Z = loss_align_Z + (Zi_prev - Zi_curr.detach()).pow(2).mean()

        K_out  = K_list[-1]
        K_idea = build_idea(labels, sigma_in=sigma_in, sigma_out=sigma_out).to(config.device)
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