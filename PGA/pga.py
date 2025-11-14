import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_cosine(X, eps=1e-8):
    X = F.normalize(input=X, eps=eps)
    sim = X @ X.t()
    return sim.clamp(min=-1.0 + eps, max=1.0 - eps)

def _knn_mask(sim, base_mask, device, topk=8):
    B = sim.size(0)
    eye = torch.eye(B, device=device)
    allowed = base_mask * (1.0 - eye)
    sim_no_diag = sim - 1e9 * eye
    masked = sim_no_diag.clone()
    masked[allowed < 0.5] = -1e9
    masked[masked < 0.0]   = -1e9

    cand = (masked > 0).sum(dim=1)
    k = min(topk, max(B - 1, 1))

    keep_row = (cand >= topk).float()
    keep_mask = keep_row[:, None] * keep_row[None, :]

    idx = torch.topk(masked, k=k, dim=1).indices

    m = torch.zeros_like(sim)
    m.scatter_(1, idx, 1.0)
    m = torch.maximum(m, m.t())
    m = m * allowed
    m = m * keep_mask
    return m


def build_intra_inter(sim, labels, device, topk=8):
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff = 1.0 - same
    M_intra = _knn_mask(sim=sim, base_mask=same, device=device, topk=topk)
    M_inter = _knn_mask(sim=sim, base_mask=diff, device=device, topk=topk)
    A_intra = (sim * M_intra).clamp(min=0.0) * same
    A_inter = (sim * M_inter).clamp(min=0.0) * diff
    return A_intra, A_inter, M_intra, M_inter

def normalize_sym(A, eps=1e-8):
    d = A.sum(dim=1).clamp_min(eps)
    Dinv = torch.diag(input=torch.pow(d, -0.5))
    return Dinv @ A @ Dinv

def diffusion_kernel(A_norm, t_diff=1):
    K = A_norm
    for _ in range(max(t_diff - 1, 0)):
        K = K @ A_norm
    return K

class GAM(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.fc2 = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.bn  = nn.BatchNorm1d(num_features=dim)

    def forward(self, A_norm, X):
        Z = A_norm @ self.fc1(X)
        Z = F.relu(input=self.bn(Z))
        Z = A_norm @ self.fc2(Z)
        return Z + X

class PGAHead(nn.Module):
    def __init__(self, num_layers, device, topk=5, t_diff=1, use_ema=True, ema_m=0.8):
        super().__init__()
        self.gams = nn.ModuleList([GAM(512) for _ in range(num_layers)])
        self.topk = topk
        self.t_diff = t_diff
        self.ema_m = ema_m
        self.device = device
        self.num_layers = num_layers
        self.use_ema = use_ema
        self.register_buffer("initialized", torch.tensor(0))
        self._ema_K = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1), requires_grad=False)
            for _ in range(num_layers)
        ])
        self.proj = nn.Linear(in_features=512, out_features=768, bias=False)
    
    def _graph(self, X, labels):
        S = pairwise_cosine(X)
        A_intra, A_inter, M_intra, M_inter = build_intra_inter(sim=S, labels=labels, device=self.device, topk=self.topk)
        A = A_intra + 1e-6 * torch.eye(S.size(0), device=self.device)
        A_norm = normalize_sym(A)
        K = diffusion_kernel(A_norm=A_norm, t_diff=self.t_diff)
        return A_norm, K, M_intra, M_inter
    
    def _init_ema_if_needed(self, K_list):
        if int(self.initialized.item()) == 1: return 
        with torch.no_grad():
            for i, K in enumerate(K_list):
                self._ema_K[i] = nn.Parameter(K.detach().clone(), requires_grad=False)
            self.initialized.fill_(1)

    def forward(self, feats_final, labels, lambda_align_K=16, lambda_align_Z=8):
        A_list, K_list, Z_list = [], [], []
        Min_list, Mout_list = [], []

        for i, (X, gam) in enumerate(zip(feats_final, self.gams)):
            Ai, Ki, Min, Mout = self._graph(X=X, labels=labels)
            Zi = gam(Ai, X)
            A_list.append(Ai); K_list.append(Ki); Z_list.append(Zi)
            Min_list.append(Min); Mout_list.append(Mout)

        if self.use_ema:
            self._init_ema_if_needed(K_list)
            with torch.no_grad():
                for i in range(self.num_layers):
                    self._ema_K[i].data.mul_(self.ema_m).add_(K_list[i].detach() * (1.0 - self.ema_m))

        def masked_mse(A, B, M, eps=1e-8):
            diff2 = (A - B).pow(2)
            num = (diff2 * M).sum()
            den = M.sum().clamp_min(eps)
            return num / den

        loss_align_K = torch.zeros((), device=self.device)
        loss_align_Z = torch.zeros((), device=self.device)

        for i in range(1, self.num_layers):
            Ki_prev = K_list[i - 1]
            Ki_tgt  = self._ema_K[i].detach() if self.use_ema else K_list[i].detach()
            M_eff = torch.maximum(Min_list[i-1], Min_list[i])
            loss_align_K = loss_align_K + masked_mse(Ki_prev, Ki_tgt, M_eff)

            # Zi_prev = F.normalize(self.proj(Z_list[i - 1]), dim=-1, eps=1e-8)
            # Zi_curr = F.normalize(self.proj(Z_list[i]),     dim=-1, eps=1e-8)

            # deg   = M_eff.sum(dim=1)
            # valid = (deg > 0).float()  
            # cos   = (Zi_prev * Zi_curr.detach()).sum(dim=-1)
            # z_i   = (1.0 - cos).clamp_min(0.0)
            # loss_align_Z += (z_i * valid).sum() / valid.sum().clamp_min(1.0)

        pairs = max(self.num_layers - 1, 1)
        loss_align_K = loss_align_K / pairs
        # loss_align_Z = loss_align_Z / pairs
        
        losses = {
            "loss_align_K": loss_align_K,
            "loss_align_Z": loss_align_Z,
            "loss_pga":     lambda_align_K * loss_align_K
                        + lambda_align_Z * loss_align_Z
        }
        return losses