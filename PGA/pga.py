import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

# ======================================================
# ==================== 基础函数区 ======================
# ======================================================

# 计算样本两两之间的余弦相似度矩阵
def pairwise_cosine(Graph, eps=1e-8):
    Graph = F.normalize(Graph, dim=-1) # [B, D]
    sim = Graph @ Graph.t()
    return sim.clamp(-1.0 + eps, 1.0 - eps) # [B, B]

# 选取 top-k 相似的邻居，sim 是相似度矩阵，base_mask 1 代表允许成为侯选边，0 代表不允许成为侯选边
def _knn_mask(sim, base_mask, topk=8):
    batch_size = sim.size(dim=0)
    # 去掉自环，不许把自己选成自己的邻居，对角线上减去了一个极大的数字，后续 topk 不会选到他
    sim_no_diag = sim - torch.eye(batch_size, device=config.device) * 1e9
    # 保持 device 一致，防止 cpu 和 cuda 混用出现问题，因为相似度可能出现负值所以这里要置为一个极小值
    # 把不允许成为侯选边的设置为极小值，不会被 topk 取到
    masked = sim_no_diag * base_mask - (1.0 - base_mask) * 1e9
    topk = min(topk, max(batch_size - 1, 1))
    # 每个维度从大到小取出前 k 个数，返回这 k 个数和他们的索引，一般用二元组承接，可以 .indices 只要索引
    idx = torch.topk(masked, k=topk, dim=1).indices
    sim_topk = torch.zeros_like(sim)
    # dim, idx, value，在 dim 维度上，将  idx 指定的位置置为 value
    sim_topk.scatter_(1, idx, 1.0)
    # 因为选取前 k 个可能得到的矩阵的单向边，这里将其变为对称矩阵
    sim_topk = torch.maximum(sim_topk, sim_topk.t())
    return sim_topk

# 构建类内类间图邻接，暂时把 topk 置为 8，Batch_size 置为 128
def build_intra_inter(sim, labels, topk=8):
    # 下面的写法经典又巧妙，.unsqueeze 在指定的维度上插入一个新维度，下面的写法利用广播得到了同类样本矩阵
    same = (labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)).float()
    diff = 1.0 - same
    sim_intra = _knn_mask(sim, same, topk) # 类内图前 topk 同类邻居
    sim_inter = _knn_mask(sim, diff, topk) # 类间图前 topk 异类邻居
    # 只保留正相似，避免噪声边，全都保留正的数值，越大越相似
    sim_intra = (sim * sim_intra).clamp(min=0.0)
    sim_inter = (sim * sim_inter).clamp(min=0.0)
    return sim_intra, sim_inter

# 做图的对称归一化，节点的度越大，信息扩散的时候就会被平均的越多，所以归一化，来平衡不同节点的影响
def normalize_sym(sim_graph, eps=1e-8):
    # D^{-1/2} A D^{-1/2}
    d = sim_graph.sum(dim=1).clamp_min(eps)
    Dinv = torch.diag(torch.pow(d, -0.5))
    return Dinv @ sim_graph @ Dinv

def diffusion_kernel(sim_base_norm, t_diff=1):
    # 和邻居信息进行融合，此时每个节点的影响范围是一跳的邻居，每循环一次就在图上多传播一层
    # 这个是先计算相似度矩阵的连乘，最后乘在节点矩阵上面进行特征的融合
    sim_diff_norm = sim_base_norm
    for _ in range(max(t_diff - 1, 0)):
        sim_diff_norm = sim_diff_norm @ sim_base_norm
    return sim_diff_norm

# 我们希望我们的相似性矩阵逼近一个理想矩阵，也就是下面构造的 Gidea，其实最理想的就是同类为 1，异类为 0，这里加上了一点点的平滑
def build_idea(labels, sigma_intra=0.99, t_diff=1):
    # 生成固定的理想图（仅用于最后一层对齐）
    same = (labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)).float()  # [B,B]
    sim_intra_idea = same * sigma_intra

    # sim_intra_idea_norm = normalize_sym(sim_intra_idea)
    # 事实证明还是不归一化比较好，选用 0.99

    K_idea = diffusion_kernel(sim_intra_idea, t_diff=t_diff)
    return K_idea

def build_idea_topk(sim_idea, labels, sigma_intra=0.99, topk=8, t_diff=1):
    # 生成固定的理想图（仅用于最后一层对齐）
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B,B]
    sim_intra = _knn_mask(sim_idea, same, topk=topk)
    sim_intra_idea = same * sigma_intra * sim_intra
    # sim_intra_idea_norm = normalize_sym(sim_intra_idea)
    # 事实证明还是不归一化比较好，选用 0.99

    K_idea = diffusion_kernel(sim_intra_idea, t_diff=t_diff)
    return K_idea
# ======================================================
# =====================GCN 模块=========================
# ======================================================
class GAM(nn.Module):
    # 传播两次的图神经网络，下一步对齐代理
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.bn  = nn.BatchNorm1d(dim)

    def forward(self, sim_graph_norm, Graph):
        Z = sim_graph_norm @ self.fc1(Graph)
        Z = F.relu(self.bn(Z))
        Z = sim_graph_norm @ self.fc2(Z)
        # 残差结构
        return Z + Graph

# ======================================================
# =====================PGA 主结构=======================
# ======================================================
class PGAHead(nn.Module):
    def __init__(self, num_layers, topk=8, t_diff=1, use_ema_target=True, ema_m=0.90):
        super().__init__()
        self.gams = nn.ModuleList([GAM(512) for _ in range(num_layers)])
        self.topk = topk
        self.t_diff = t_diff
        self.num_layers = num_layers

        # α, β 可为标量或每层独立的 list
        self.alpha_sched = torch.linspace(1.00, 1.20, steps=num_layers).tolist()
        self.beta_sched  = torch.linspace(0.00, 0.00, steps=num_layers).tolist()

        # EMA 缓冲区初始化了 EMA 的占位符等信息，现在还没开始真正的存储平滑图
        # 我们不直接拿当前的 batch 算出来的 K 对齐，而是对他做一个平滑
        # 是否启用 EMA 平滑，方便打开和关闭
        self.use_ema_target = use_ema_target
        self.ema_m = ema_m
        # 初始化标志，初始化代表没建立平均图
        self.register_buffer("initialized", torch.tensor(0))
        self._ema_K = nn.ParameterList([
            # 存放每层的平均图结构，保证张量随着设备移动，占位符 (1, 1) 代表还没拿到真实的 K
            nn.Parameter(torch.zeros(1, 1), requires_grad=False)
            # requires_grad 表示这些不是可学习的参数只是简单的缓存
            for _ in range(num_layers)
        ])

        # 投影层，将每层 Z 投影到统一 768 维空间以进行语义对齐
        self.proj = nn.Linear(512, 768, bias=False)
    # 可以在外面调整调度表
    def set_alpha_beta(self, alpha, beta):
        self.alpha_sched = alpha
        self.beta_sched = beta

    def _get_alpha_beta(self, layer_idx):
        a = torch.tensor(self.alpha_sched[layer_idx], device=config.device, dtype=torch.float32)
        b = torch.tensor(self.beta_sched[layer_idx], device=config.device, dtype=torch.float32)
        return a, b

    def _graph(self, Graph, labels, a, b):
        sim = pairwise_cosine(Graph)
        sim_intra, sim_inter = build_intra_inter(sim, labels=labels, topk=self.topk)
        # 加一点自环，避免度为 0
        sim_graph = a * sim_intra + b * sim_inter + 1e-6 * torch.eye(sim.size(dim=0), device=config.device)
        sim_graph_norm = normalize_sym(sim_graph=sim_graph)
        K = diffusion_kernel(sim_base_norm=sim_graph_norm, t_diff=self.t_diff)
        return sim_graph_norm, K

    def _init_ema_if_needed(self, K_list):
        # 开始跑到 时候创建副本，后续不再进行更改
        if int(self.initialized.item()) == 1: return
        # 不记录梯度，因为只是简单的复制数据
        # 遍历每层的真实图 K 从计算图里面分离并复制一份不会参与反向传播，存储值副本
        with torch.no_grad(): 
            for i, K in enumerate(K_list):
                self._ema_K[i] = nn.Parameter(K.detach().clone(), requires_grad=False)
            self.initialized.fill_(1)
    
    def forward(self, feats_final, labels, lambda_align_K=64, lambda_align_Z=16, sigma_intra=0.99):

        sim_graph_norm_list, K_list, Z_list = [], [], []
        for i, (Graph, gam) in enumerate(zip(feats_final, self.gams)):
            # 取出 alpha、beta
            a_i, b_i = self._get_alpha_beta(layer_idx=i)
            # 计算扩散核和原始核
            sim_graph_norm, K = self._graph(Graph=Graph, labels=labels, a=a_i, b=b_i)
            # 扩散核计算 GCN 融合邻居信息
            Z = gam(sim_graph_norm, Graph)
            sim_graph_norm_list.append(sim_graph_norm); K_list.append(K); Z_list.append(Z) 

        if self.use_ema_target:
            self._init_ema_if_needed(K_list=K_list)
            with torch.no_grad():
                for i in range(self.num_layers):
                    # 这个只是 detach 就行只是用了一下，没有修改 K 的值
                    self._ema_K[i].data.mul_(self.ema_m).add_(K_list[i].detach() * (1.0 - self.ema_m))

        # 逐层定向对齐：|| K_{i-1} - sg[K_i] ||^2 和 || Z_{i-1} - sg[Z_i] ||^2
        loss_align_K = torch.zeros((), device=config.device)
        loss_align_Z = torch.zeros((), device=config.device)
        for i in range(1, self.num_layers):
            K_pre = K_list[i - 1]
            K_tgt  = self._ema_K[i].detach() if self.use_ema_target else K_list[i].detach()
            loss_align_K = loss_align_K + (K_pre - K_tgt).pow(2).mean()

            Z_pre = F.normalize(self.proj(Z_list[i - 1]), dim=-1)
            Z_cur = F.normalize(self.proj(Z_list[i]), dim=-1)
            loss_align_Z = loss_align_Z + (Z_pre - Z_cur.detach()).pow(2).mean()

        K_out  = K_list[-1]

        K_idea = build_idea(labels=labels, sigma_intra=sigma_intra, t_diff=self.t_diff).to(config.device)
        # sim_idea = pairwise_cosine(Graph=feats_final[-1])
        # K_idea = build_idea_topk(sim_idea=sim_idea, labels=labels, sigma_intra=0.99, topk=self.topk, t_diff=1)
        loss_idea = (K_out - K_idea).pow(2).mean()

        losses = {
            # loss_idea 不宜太大，因为他的对比的绝对参照物，和前面的图与图之间的结构有很大的不同
            "loss_pga": lambda_align_K * loss_align_K + lambda_align_Z * loss_align_Z + loss_idea
        }
        return losses