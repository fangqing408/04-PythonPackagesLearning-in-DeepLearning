import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def pairwise_cosine(X, eps=1e-8):
    '''
    input: 
    X: [B, 512]
    output: 
    sim: [B, B]
    '''
    X = F.normalize(X, eps=eps) # [B, 512]
    sim = X @ X.t() # [B, B]
    return sim.clamp(min=-1.0+eps, max=1.0-eps) # [B, B]

def _knn_mask(sim, base_mask, device, topk=5):
    '''
    input: 
    sim: [B, B]
    base_mask: [B, B]
    output: 
    M: [B, B]
    '''
    B = sim.size(0)
    eye = torch.eye(B, device=device) # [B, B]
    allowed = base_mask * (1.0 - eye) # [B, B]
    sim_no_diag = sim - 1e9 * eye # [B, B]
    masked = sim_no_diag.clone() # [B, B]
    masked[allowed < 0.5] = -1e9 # [B, B]
    masked[masked < 0.0] = -1e9 # [B, B]
    cand = (masked > 0).sum(dim=1) # [B]
    k = min(topk, max(B - 1, 1))
    keep_row = (cand >= topk).float() # [B]
    keep_mask = keep_row[:, None] * keep_row[None, :] # [B, B]
    idx = torch.topk(masked, k=k, dim=1).indices  # [B, k]
    M = torch.zeros_like(sim) # [B, B]
    M.scatter_(1, idx, 1.0) # [B, B]
    M = torch.maximum(M, M.t()) # [B, B]
    M = M * allowed # [B, B]
    M = M * keep_mask # [B, B]
    return M # [B, B]

def build_intra(sim, labels, device, topk=5):
    '''
    input: 
    sim: [B, B]
    labels: [B]
    output: 
    A_intra: [B, B]
    M_intra: [B, B]
    '''
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
    M_intra = _knn_mask(sim=sim, base_mask=same, device=device, topk=topk) # [B, B]
    A_intra = (sim * M_intra).clamp(min=0.0) * same # [B, B]
    return A_intra, M_intra # [B, B]

def normalize_sym(A, eps=1e-8):
    '''
    input: 
    A: [B, B]
    output: 
    [B, B]
    '''
    d = A.sum(dim=1).clamp_min(eps)
    Dinv = torch.diag(torch.pow(d, -0.5))
    return Dinv @ A @ Dinv

def diffusion_kernel(A_norm, t_diff=1):
    '''
    input: 
    A_norm: [B, B]
    output: 
    K: [B, B]
    '''
    K = A_norm
    for _ in range(max(t_diff - 1, 0)):
        K = K @ A_norm
    return K

class IntraClassEMAMemory:
    '''
    bank[label][sample_id] = score (float)
    '''
    def __init__(self, momentum=0.9, default=0.5):
        self.m = momentum
        self.default = default
        self.bank = defaultdict(dict)  # {label: {sample_id: score}}

    @torch.no_grad()
    def update(self, labels, sample_ids, scores):
        '''
        input:
        labels: [B]
        sample_ids: [B]
        scores: [B]
        '''
        labels = labels.detach().cpu().tolist()
        sample_ids = sample_ids.detach().cpu().tolist()
        scores = scores.detach().cpu().tolist()
        for y, sid, s in zip(labels, sample_ids, scores):
            prev = self.bank[y].get(sid, self.default)
            new  = self.m * prev + (1.0 - self.m) * s
            self.bank[y][sid] = float(new)

    @torch.no_grad()
    def fetch(self, labels, sample_ids):
        '''
        input:
        lables: [B]
        sample_ids: [B]
        output:
        [B]
        '''
        device = labels.device
        labels_cpu = labels.detach().cpu().tolist()
        sample_ids_cpu = sample_ids.detach().cpu().tolist()
        out = []
        for y, sid in zip(labels_cpu, sample_ids_cpu):
            out.append(self.bank[y].get(sid, self.default))
        return torch.tensor(out, device=device, dtype=torch.float32)

class PGAHead(nn.Module):
    def __init__(self, num_layers, device, topk=5, t_diff=1, memory_m=0.9):
        super().__init__()
        self.num_layers = num_layers
        self.device = device
        self.topk = topk
        self.t_diff = t_diff

        self.intra_memories = [
            IntraClassEMAMemory(momentum=memory_m)
            for _ in range(num_layers)
        ]

    def forward(self, feats_final, labels, sample_ids, lambda_align_K=16.0):
        '''
        input:
        feats_final: [num_layers, B, 512]
        labels: [B]
        sample_ids: [B]
        '''
        device = labels.device
        B = labels.size(0)
        debug = {}
        # PK Subset
        with torch.no_grad():
            X_last = feats_final[-1] # [B, 512]
            S_last_full = pairwise_cosine(X_last) # [B, B]
            _, M_intra_last_full = build_intra(sim=S_last_full, labels=labels, device=self.device, topk=self.topk) # [B, B]
            cand_mask = (M_intra_last_full > 0).sum(dim=1) > 0 # [B]
            if cand_mask.sum() < 2:
                zero = torch.zeros((), device=device)
                return {
                    "loss_align_K": zero,
                    "loss_pga": lambda_align_K * zero,
                        "debug": {
                        "B": int(B),
                        "b_sub": 0,
                        "sub_ratio": 0.0,
                    }
                }
            idx_sub = cand_mask.nonzero(as_tuple=True)[0] # [b]
            labels_sub = labels[idx_sub] # [b]
            sample_ids_sub = sample_ids[idx_sub] # [b]

        b = labels_sub.size(0)
        debug["B"] = int(B)
        debug["b_sub"] = int(b)
        debug["sub_ratio"] = float(b / float(B))

        K_sub_list, Min_sub_list, S_sub_list = [], [], [] # [b, b]
        eye_b = torch.eye(b, device=device)
        for i in range(self.num_layers):
            X_i_sub = feats_final[i][idx_sub] # [b, 512]
            S_i_sub = pairwise_cosine(X_i_sub) # [b, b]
            A_intra_sub, M_intra_sub = build_intra(sim=S_i_sub, labels=labels_sub, device=self.device, topk=self.topk) # [b, b]

            # A_sub = A_intra_sub + 1e-6 * eye_b
            # A_norm_sub = normalize_sym(A_sub)
            # K_sub = diffusion_kernel(A_norm_sub, t_diff=self.t_diff)

            K_sub = A_intra_sub # [b, b]

            S_sub_list.append(S_i_sub) # [b, b]
            Min_sub_list.append(M_intra_sub) # [b, b]
            K_sub_list.append(K_sub) # [b, b]

        ema_scores = [] 
        main_ratios = [] 

        with torch.no_grad():
            same_sub = (labels_sub[:, None] == labels_sub[None, :]).float() # [b, b]
            for i in range(self.num_layers):
                S_i_sub = S_sub_list[i] # [b, b]
                S_intra_i = (S_i_sub * same_sub).clamp_min(0.0) # [b, b]

                deg = (S_intra_i > 0).float().sum(dim=1).clamp_min(1.0) # [b]
                score_cur = S_intra_i.sum(dim=1) / deg # [b]
                score_cur = torch.sigmoid(score_cur) # [b]

                mem = self.intra_memories[i]
                mem.update(labels_sub, sample_ids_sub, score_cur)
                score_ema = mem.fetch(labels_sub, sample_ids_sub) # [b]
                
                main_flag_i = (score_ema >= 0.5).float() # [b]

                ema_scores.append(score_ema) 
                main_ratios.append(float(main_flag_i.mean().item()))

        debug["main_ratio_per_layer"] = main_ratios

        def masked_mse(A, B, M, W=None, eps=1e-8):
            '''
            input:
            A: [b, b]
            B: [b, b]
            '''
            diff2 = (A - B).pow(2)
            if W is not None:
                diff2 = diff2 * W
            num = (diff2 * M).sum()
            den = (M * (W if W is not None else 1.0)).sum().clamp_min(eps)
            return num / den

        loss_align_K = torch.zeros((), device=device)
        eff_edges_per_pair = [] 

        for i in range(1, self.num_layers):
            Ki_deep = K_sub_list[i] # [b, b]
            Ki_tgt  = K_sub_list[i - 1].detach() # [b, b]  

            Min_prev = Min_sub_list[i - 1]
            Min_curr = Min_sub_list[i]

            M_eff = torch.maximum(Min_prev, Min_curr)

            num_eff_edges = int(M_eff.sum().item())
            eff_edges_per_pair.append(num_eff_edges)

            if num_eff_edges > 0:
                score_ema_prev = ema_scores[i - 1] # [b]
                W_prev = score_ema_prev[:, None] * score_ema_prev[None, :]
                loss_align_K = loss_align_K + masked_mse(Ki_deep, Ki_tgt, M_eff, W=W_prev)
        pairs = max(self.num_layers - 1, 1)
        loss_align_K = loss_align_K / pairs

        debug["eff_edges_per_pair"] = eff_edges_per_pair
        if len(eff_edges_per_pair) > 0:
            debug["eff_edges_mean"] = float(
                torch.tensor(eff_edges_per_pair, dtype=torch.float32).mean().item()
            )
        else:
            debug["eff_edges_mean"] = 0.0

        debug["loss_align_K"] = float(loss_align_K.item())

        losses = {
            "raw_pga": loss_align_K,
            "loss_pga": lambda_align_K * loss_align_K,
            "debug": debug
        }
        return losses