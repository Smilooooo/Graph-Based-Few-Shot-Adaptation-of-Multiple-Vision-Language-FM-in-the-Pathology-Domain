"""
CrossFM Transductive Instance Graph — Multi-View Cosine (CTFIG-MVC) Model.

Combines CTFIG-MV (every augmented view = separate graph node) with a cosine
classification head instead of the MLP used in CTFIG/CTFIG-MV.

Supports a USE_TRANSDUCTIVE flag to ablate whether putting the query in the graph
helps or hurts. When use_transductive=False the model behaves like CFIG but with
multi-view support nodes — query is classified by cosine without graph refinement.

Forward pass (use_transductive=True):
    1. V_all = cat([V_inst, V_query], dim=1)        transductive
    2. V_all_refined, T_refined = CFIG(V_all, T_raw)
    3. Split V_inst_refined [K, C*N*n_ep, D] and Q_for_classify = V_all_refined[:, CN:]

Forward pass (use_transductive=False):
    1. V_inst_refined, T_refined = CFIG(V_inst, T_raw)   support only
    2. Q_for_classify = V_query                            raw query, not in graph

Both paths then:
    4. V_raw_mean = V_inst.view(K,C,-1,D).mean(2)
       V_proto    = V_inst_refined.view(K,C,-1,D).mean(2)
    5. V_final = sigmoid(alpha_v)*V_proto + (1-sigmoid(alpha_v))*V_raw_mean
       T_final = sigmoid(alpha_t)*T_refined + (1-sigmoid(alpha_t))*T_raw
    6. per-FM cosine: proto_k = sigmoid(gate_vt)*V_final[k] + (1-sigmoid(gate_vt))*T_final[k]
       logits = mean over K FMs of exp(logit_scale)*norm(Q_for_classify[k])@norm(proto_k).T

Graph code (CFIGLayer, CFIG) is inlined from cfig_model.py to decouple from
other trainers and support CTFIG-MVC-specific ablations (e.g. shared cross-modal W).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CFIGLayer -- one round of heterogeneous message passing
# =============================================================================

class CFIGLayer(nn.Module):
    """
    One round of heterogeneous message passing over the cross-FM instance graph.

    Operates on:
        V_inst : [K, C*N, D]  individual support sample features
        T      : [K, C,   D]  class-level text prototypes

    Six edge types (shared weight matrices across FMs):
        W_VV : intra-FM instance->instance  (cosine adj, top-k, sym-norm)
        W_TT : intra-FM text->text          (cosine adj, top-k, sym-norm)
        W_TV : intra-FM text->instance      (rectangular row-norm affinity)
        W_VT : intra-FM instance->text      (rectangular col-norm affinity)
        W_xV : cross-FM instance->instance  (mean of same-instance from other K-1 FMs)
        W_xT : cross-FM text->text          (mean of same-class text, identical to CFPG)

    When share_crossmodal_w=True, W_VT and W_TV are the same nn.Linear
    (saves 262K params; ablation for whether two separate matrices are needed).

    When crossfm_attention=True, cross-FM aggregation uses GAT-style learned
    attention instead of uniform mean over the K-1 neighbor FMs.
    """

    def __init__(
        self,
        dim: int,
        top_k: int = 3,
        dropout: float = 0.1,
        gate_cv_init: float = 0.0,
        gate_ct_init: float = 0.0,
        gate_xv_init: float = 0.0,
        gate_xt_init: float = 0.0,
        use_intramodal: bool = True,
        use_crossmodal: bool = True,
        use_crossfm: bool = True,
        share_crossmodal_w: bool = False,
        crossfm_attention: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.use_intramodal = use_intramodal
        self.use_crossmodal = use_crossmodal
        self.use_crossfm = use_crossfm
        self.crossfm_attention = crossfm_attention

        self.W_VV = nn.Linear(dim, dim, bias=False)
        self.W_TT = nn.Linear(dim, dim, bias=False)
        self.W_TV = nn.Linear(dim, dim, bias=False)
        if share_crossmodal_w:
            self.W_VT = self.W_TV
        else:
            self.W_VT = nn.Linear(dim, dim, bias=False)
        self.W_xV = nn.Linear(dim, dim, bias=False)
        self.W_xT = nn.Linear(dim, dim, bias=False)

        if crossfm_attention:
            self.attn_xV_l = nn.Parameter(torch.empty(dim))
            self.attn_xV_r = nn.Parameter(torch.empty(dim))
            self.attn_xT_l = nn.Parameter(torch.empty(dim))
            self.attn_xT_r = nn.Parameter(torch.empty(dim))
            nn.init.xavier_uniform_(self.attn_xV_l.unsqueeze(0))
            nn.init.xavier_uniform_(self.attn_xV_r.unsqueeze(0))
            nn.init.xavier_uniform_(self.attn_xT_l.unsqueeze(0))
            nn.init.xavier_uniform_(self.attn_xT_r.unsqueeze(0))
            self.leaky_relu = nn.LeakyReLU(0.2)

        self.gate_cv_raw = nn.Parameter(torch.tensor(float(gate_cv_init)))
        self.gate_ct_raw = nn.Parameter(torch.tensor(float(gate_ct_init)))
        self.gate_xv_raw = nn.Parameter(torch.tensor(float(gate_xv_init)))
        self.gate_xt_raw = nn.Parameter(torch.tensor(float(gate_xt_init)))

        self.norm_V = nn.LayerNorm(dim)
        self.norm_T = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    # ── Adjacency helpers ───────────────────────────────────────────────────

    def _cosine_adj_batched(self, x: torch.Tensor) -> torch.Tensor:
        K, N, _ = x.shape
        x_norm = F.normalize(x.detach(), dim=-1)
        sim    = torch.bmm(x_norm, x_norm.transpose(-1, -2))

        eye = torch.eye(N, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        sim = (sim * (1.0 - eye)).clamp(min=0.0)

        if self.top_k > 0 and self.top_k < N - 1:
            _, topk_idx = sim.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(sim, dtype=torch.bool)
            bk = torch.arange(K, device=sim.device).view(K, 1, 1).expand(K, N, self.top_k)
            ri = torch.arange(N, device=sim.device).view(1, N, 1).expand(K, N, self.top_k)
            mask[bk, ri, topk_idx] = True
            mask = mask | mask.transpose(-1, -2)
            sim  = sim * mask.float()

        sim = sim + eye
        d   = sim.sum(dim=-1).clamp(min=1e-8).pow(-0.5)
        return sim * d.unsqueeze(-1) * d.unsqueeze(-2)

    def _cross_affinity_rect(
        self, v_inst: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K, CN, _ = v_inst.shape
        C = t.size(1)

        v_norm = F.normalize(v_inst.detach(), dim=-1)
        t_norm = F.normalize(t.detach(), dim=-1)
        S      = torch.bmm(v_norm, t_norm.transpose(-1, -2)).clamp(min=0.0)

        if self.top_k > 0 and self.top_k < C:
            _, topk_idx = S.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(S, dtype=torch.bool)
            bk = torch.arange(K, device=S.device).view(K, 1, 1).expand(K, CN, self.top_k)
            ri = torch.arange(CN, device=S.device).view(1, CN, 1).expand(K, CN, self.top_k)
            mask[bk, ri, topk_idx] = True
            S = S * mask.float()

        row_sum = S.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        S_row   = S / row_sum

        col_sum = S.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        S_VT    = (S / col_sum).transpose(-1, -2)

        return S_row, S_VT

    # ── Cross-FM GAT attention ─────────────────────────────────────────────

    def _crossfm_gat(
        self,
        X: torch.Tensor,          # [K, N, D]
        W: nn.Linear,             # transform
        attn_l: torch.Tensor,     # [D] source attention vector
        attn_r: torch.Tensor,     # [D] neighbor attention vector
    ) -> torch.Tensor:
        """GAT attention over same-node features across K FMs. Returns [K, N, D]."""
        K = X.size(0)
        H = W(X)                                              # [K, N, D]
        score_l = (H * attn_l).sum(-1)                        # [K, N]
        score_r = (H * attn_r).sum(-1)                        # [K, N]
        e = self.leaky_relu(
            score_l.unsqueeze(1) + score_r.unsqueeze(0)       # [K, K, N]
        )
        mask = torch.eye(K, device=e.device, dtype=torch.bool).unsqueeze(-1)
        e = e.masked_fill(mask, float('-inf'))
        alpha = F.softmax(e, dim=1)                           # [K, K, N]
        return torch.einsum('ijk,jkd->ikd', alpha, H)         # [K, N, D]

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        V_inst: torch.Tensor,
        T:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K = V_inst.size(0)

        if self.use_intramodal:
            A_VV = self._cosine_adj_batched(V_inst)
            A_TT = self._cosine_adj_batched(T)

        if self.use_crossmodal:
            S_TV, S_VT = self._cross_affinity_rect(V_inst, T)

        if self.use_intramodal:
            m_VV = torch.bmm(A_VV, self.W_VV(V_inst))
            m_TT = torch.bmm(A_TT, self.W_TT(T))
        else:
            m_VV = torch.zeros_like(V_inst)
            m_TT = torch.zeros_like(T)

        if self.use_crossmodal:
            m_TV = torch.bmm(S_TV, self.W_TV(T))
            m_VT = torch.bmm(S_VT, self.W_VT(V_inst))
        else:
            m_TV = torch.zeros_like(V_inst)
            m_VT = torch.zeros_like(T)

        if self.use_crossfm and K > 1:
            if self.crossfm_attention:
                m_xV = self._crossfm_gat(V_inst, self.W_xV, self.attn_xV_l, self.attn_xV_r)
                m_xT = self._crossfm_gat(T,      self.W_xT, self.attn_xT_l, self.attn_xT_r)
            else:
                V_xfm = (V_inst.sum(0, keepdim=True) - V_inst) / (K - 1)
                T_xfm = (T.sum(0, keepdim=True) - T) / (K - 1)
                m_xV  = self.W_xV(V_xfm)
                m_xT  = self.W_xT(T_xfm)
        else:
            m_xV = torch.zeros_like(V_inst)
            m_xT = torch.zeros_like(T)

        gate_cv = torch.sigmoid(self.gate_cv_raw)
        gate_ct = torch.sigmoid(self.gate_ct_raw)
        gate_xv = torch.sigmoid(self.gate_xv_raw)
        gate_xt = torch.sigmoid(self.gate_xt_raw)

        agg_V = m_VV + gate_cv * m_TV + gate_xv * m_xV
        agg_T = m_TT + gate_ct * m_VT + gate_xt * m_xT

        V_new = self.norm_V(V_inst + self.dropout(torch.tanh(agg_V)))
        T_new = self.norm_T(T      + self.dropout(torch.tanh(agg_T)))

        return V_new, T_new


# =============================================================================
# CFIG -- stack of CFIGLayer
# =============================================================================

class CFIG(nn.Module):

    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        top_k: int = 3,
        dropout: float = 0.1,
        gate_cv_init: float = 0.0,
        gate_ct_init: float = 0.0,
        gate_xv_init: float = 0.0,
        gate_xt_init: float = 0.0,
        use_intramodal: bool = True,
        use_crossmodal: bool = True,
        use_crossfm: bool = True,
        share_crossmodal_w: bool = False,
        crossfm_attention: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CFIGLayer(
                dim=dim, top_k=top_k, dropout=dropout,
                gate_cv_init=gate_cv_init, gate_ct_init=gate_ct_init,
                gate_xv_init=gate_xv_init, gate_xt_init=gate_xt_init,
                use_intramodal=use_intramodal,
                use_crossmodal=use_crossmodal,
                use_crossfm=use_crossfm,
                share_crossmodal_w=share_crossmodal_w,
                crossfm_attention=crossfm_attention,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self, V_inst: torch.Tensor, T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            V_inst, T = layer(V_inst, T)
        return V_inst, T


# =============================================================================
# CrossFMTransductiveInstanceGraphMVCModel -- full pipeline
# =============================================================================

class CrossFMTransductiveInstanceGraphMVCModel(nn.Module):
    """
    Full CTFIG-MVC pipeline: multi-view transductive graph + cosine head.

    Learnable parameters (4 scalars, or 4+K with learned FM weights):
        alpha_v_raw   : residual blend weight for visual prototype
        alpha_t_raw   : residual blend weight for text prototype
        gate_vt_raw   : blend weight between V_final and T_final per FM
        logit_scale   : temperature (exp-clamped to 100)
        fm_logits_raw : (optional) per-FM softmax weights for logit aggregation
    """

    def __init__(
        self,
        num_fms:          int,
        feat_dim:         int,
        num_classes:      int,
        num_gnn_layers:   int   = 1,
        top_k:            int   = 3,
        dropout:          float = 0.1,
        alpha_v_init:     float = 0.5,
        alpha_t_init:     float = 0.5,
        gate_vt_init:     float = 0.0,
        logit_scale_init: float = 2.3,
        gate_cv_init:     float = 0.0,
        gate_ct_init:     float = 0.0,
        gate_xv_init:     float = 0.0,
        gate_xt_init:     float = 0.0,
        use_intramodal:   bool  = True,
        use_crossmodal:   bool  = True,
        use_crossfm:      bool  = True,
        use_transductive: bool  = True,
        share_crossmodal_w: bool = False,
        crossfm_attention: bool = False,
        use_learned_fm_weights: bool = False,
        use_prototype_nodes: bool = False,
    ) -> None:
        super().__init__()
        self.num_fms        = num_fms
        self.num_classes    = num_classes
        self.use_transductive = use_transductive
        self.use_learned_fm_weights = use_learned_fm_weights
        self.use_prototype_nodes = use_prototype_nodes

        self.instance_graph = CFIG(
            dim=feat_dim, num_layers=num_gnn_layers, top_k=top_k, dropout=dropout,
            gate_cv_init=gate_cv_init, gate_ct_init=gate_ct_init,
            gate_xv_init=gate_xv_init, gate_xt_init=gate_xt_init,
            use_intramodal=use_intramodal,
            use_crossmodal=use_crossmodal,
            use_crossfm=use_crossfm,
            share_crossmodal_w=share_crossmodal_w,
            crossfm_attention=crossfm_attention,
        )

        def _logit(v: float) -> float:
            v = max(min(v, 1.0 - 1e-6), 1e-6)
            return math.log(v / (1.0 - v))

        self.alpha_v_raw = nn.Parameter(torch.tensor(_logit(alpha_v_init)))
        self.alpha_t_raw = nn.Parameter(torch.tensor(_logit(alpha_t_init)))
        self.gate_vt_raw = nn.Parameter(torch.tensor(float(gate_vt_init)))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

        if use_learned_fm_weights:
            self.fm_logits_raw = nn.Parameter(torch.zeros(num_fms))

    def forward(
        self,
        query_features:    List[torch.Tensor],
        vsupport_features: List[torch.Tensor],
        tproto_features:   List[torch.Tensor],
    ) -> torch.Tensor:
        """Returns logits [B, C]."""
        K = self.num_fms
        C = self.num_classes

        V_inst  = torch.stack(vsupport_features, dim=0)
        T_raw   = torch.stack(tproto_features,   dim=0)
        V_query = torch.stack(query_features,    dim=0)

        CN = V_inst.size(1)
        D  = V_inst.size(2)

        if self.use_prototype_nodes:
            V_inst = V_inst.view(K, C, -1, D).mean(dim=2)
            CN = C

        V_raw_mean = V_inst.view(K, C, -1, D).mean(dim=2)

        if self.use_transductive:
            V_all = torch.cat([V_inst, V_query], dim=1)
            V_all_refined, T_refined = self.instance_graph(V_all, T_raw)
            V_inst_refined  = V_all_refined[:, :CN, :]
            Q_for_classify  = V_all_refined[:, CN:, :]
        else:
            V_inst_refined, T_refined = self.instance_graph(V_inst, T_raw)
            Q_for_classify  = V_query

        V_proto = V_inst_refined.view(K, C, -1, D).mean(dim=2)

        alpha_v = torch.sigmoid(self.alpha_v_raw)
        alpha_t = torch.sigmoid(self.alpha_t_raw)
        V_final = alpha_v * V_proto   + (1.0 - alpha_v) * V_raw_mean
        T_final = alpha_t * T_refined + (1.0 - alpha_t) * T_raw

        gate_vt = torch.sigmoid(self.gate_vt_raw)
        scale   = self.logit_scale.exp().clamp(max=100.0)

        logits_list = []
        for k in range(K):
            proto_k = gate_vt * V_final[k] + (1.0 - gate_vt) * T_final[k]
            q_k = F.normalize(Q_for_classify[k], dim=-1)
            p_k = F.normalize(proto_k,           dim=-1)
            logits_list.append(scale * (q_k @ p_k.T))

        stacked = torch.stack(logits_list, dim=0)
        if self.use_learned_fm_weights:
            w = F.softmax(self.fm_logits_raw, dim=0)
            return (stacked * w.view(K, 1, 1)).sum(dim=0)
        return stacked.mean(dim=0)
