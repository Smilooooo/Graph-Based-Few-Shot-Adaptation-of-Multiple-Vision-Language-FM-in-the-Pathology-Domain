"""
MLPHeGraphAdapter Model – Early MLP Fusion + HeGraph Heterogeneous GCN.

Architecture
────────────
Stage 1  Early Fusion + JointAdapter (identical to MLPGraphAdapter)
    Concatenate features from all 3 frozen FMs:
        [C, 512] × 3  →  cat  →  [C, 1536]
    Project to common space via shared MLP:
        JointAdapter: [N, 1536]  →  [N, 512]
    Applied identically to: query, positive text protos, negative text protos,
    visual class-mean protos, per-sample Tip-Adapter cache.

Stage 2  HeGraph Heterogeneous GCN  (Sec 3.1 of He et al., 2024)
    Three node types in the fused 512-dim space:
        P — positive text prototypes  [C, 512]
        N — negative text prototypes  [C, 512]
        V — visual class-mean protos  [C, 512]
    Six adjacency matrices (all computed from detached adapted features):
        P→N (diagonal negated), V→N (diagonal negated),
        P→P (self-loops removed),  V→P,  N→P (diagonal negated),  V→V (self-loops removed)
    Three sequential update steps (fixed meta-path weights, paper defaults):
        Step 1 (Eq. 1-2):  N ← N + β_p2n·GCN(N,P) + β_v2n·GCN(N,V)
        Step 2 (Eq. 3-4):  P ← P + α_p2p·GCN(P,P) + α_v2p·GCN(P,V) + α_n2p·GCN(P,N_updated)
        Step 3 (Eq. 5-6):  vis_bias ← GCN(V,V)  [no self-add; used to update Tip-Adapter cache]
    Note: α_n2p differs at train (0.2) vs test (0.1).

Stage 3  Dual Classification
    Text logits (Eq. 7):   τ · query_norm @ P_refined_norm.T          [B, C]
    Visual logits (Eq. 8): τ · exp(sim(query, cache_final) − 1) @ L   [B, C]
        where cache_final = cache_adapted + γ · (L @ vis_bias)
    Test output: (softmax(L_text) + w · softmax(L_cache)) / (1 + w)
        with w = cache_output hyperparameter.

Departures from single-backbone HeGraph
────────────────────────────────────────
1. JointAdapter fuses 3 FM features; adjacency built from adapted (detached) features
   rather than frozen base features — graph structure evolves with the adapter.
2. Learnable logit_scale (nn.Parameter) replaces frozen CLIP logit_scale.
3. Cache and all prototypes pass through JointAdapter at every forward pass so they
   stay in the same metric space as the adapter trains.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.mlpga_model import JointAdapter


# =============================================================================
# Helpers (adapted from he_graph_paper.py — He et al., 2024)
# =============================================================================

def _compute_adj(x: torch.Tensor, y: torch.Tensor,
                 negative: bool = False) -> torch.Tensor:
    """
    Compute a cosine-similarity adjacency matrix.

    Parameters
    ──────────
    x, y : [1, N, D]  node feature matrices (will be L2-normalised)
    negative : if True, negate only the diagonal entries (same-class entries)
               while keeping off-diagonal positive — models 'opposite class'
               information for N-type adjacency (Sec 3.1.1).

    Returns
    ───────
    A : [1, N, N]  raw cosine-similarity adjacency in [-1, 1] (diagonal negated
                   when negative=True). No clamping: the paper (Sec 3.1.1) uses
                   the cosine similarity directly as the edge weight, so negative
                   (anti-correlated) edges are kept. The GCN's degree term uses
                   abs(adj), so signed edges are handled correctly.
    """
    xn = F.normalize(x, p=2, dim=2)
    yn = F.normalize(y, p=2, dim=2)
    A  = torch.bmm(xn, yn.transpose(1, 2))    # [1, N, N]

    if negative:
        mask = torch.eye(A.shape[1], A.shape[2], device=A.device).unsqueeze(0)
        A = (1.0 - mask) * A + mask * (-A)
    return A


class _GraphModel(nn.Module):
    """
    Paper-faithful single-weight GCN layer (Eq. 1 / 3 / 5 of He et al., 2024).

    Text nodes    (Eq. 1 / 3, self_add=True):
        m_i = tanh( 1/(d_i+1) · W·x_i  +  Σ_j  r_ij / √((d_i+1)(d_j+1)) · W·x_j )
    Visual nodes  (Eq. 5, self_add=False)  — the self-loop term is DROPPED:
        m_i = tanh(                        Σ_j  r_ij / √((d_i+1)(d_j+1)) · W·x_j )

    So the visual-aggregation call (self_add=False) realises Eq. 5, whose output is
    a per-class *bias* added to the Tip-Adapter cache (Eq. 6), not a node update.
    The two text calls (self_add=True) realise Eq. 1 / 3 and residual-update the node.

    Activation: tanh (not ReLU). HeGraphAdapter extends GraphAdapter, whose
    GCN uses tanh, and the HeGraphAdapter paper does not specify the activation.
    tanh is also sign-preserving, so it keeps the dissimilarity signal carried
    by the sign-flipped edges into the negative-text nodes, which ReLU would clip.
    """

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.W   = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        stdv = 1.0 / math.sqrt(hidden_dim)
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                adj: torch.Tensor, self_add: bool = True) -> torch.Tensor:
        """
        Parameters
        ──────────
        x   : [1, N, D]  self-feature (node being updated)
        y   : [1, N, D]  neighbour features
        adj : [1, N, N]  edge weights r_{ij}
        self_add : whether to include the (1/d_i)·W·x_i self-loop term
        """
        degree    = torch.abs(adj).sum(-1, keepdim=True) + 1     # [1,N,1]
        sqrt_deg  = torch.sqrt(degree)
        d_denom   = torch.bmm(sqrt_deg, sqrt_deg.transpose(1, 2))  # [1,N,N]

        Wy             = torch.matmul(y, self.W)                   # [1,N,D]
        neighbour_agg  = torch.bmm(adj / d_denom, Wy)             # [1,N,D]

        if self_add:
            Wx = torch.matmul(x, self.W)
            output = (1.0 / degree) * Wx + neighbour_agg
        else:
            output = neighbour_agg

        return self.act(output)                                    # [1,N,D]


# =============================================================================
# HeGraphLearnerMFM – heterogeneous GCN on pre-adapted features
# =============================================================================

class HeGraphLearnerMFM(nn.Module):
    """
    HeGraph 3-step message passing operating on already-adapted 512-dim features.

    All input features arrive from JointAdapter (live in a shared metric space).
    Adjacency matrices are computed from detached copies of the adapted features
    so the graph structure contributes no gradient — only the GCN weight matrices
    (W^n, W^p, W^v) and node features provide gradients.

    Meta-path weights are fixed (not learnable), matching the paper's Table 3
    finding that fixed weights outperform learnable ones.

    Parameters
    ──────────
    hidden_dim       : node feature dimension (= JointAdapter output dim, 512)
    num_classes      : number of classes C
    beta_p2n/v2n     : fixed weights for N-node update (Eq. 2)
    alpha_p2p/v2p    : fixed weights for P-node update (Eq. 4)
    alpha_n2p_train  : α_{n→p} used at training time  (Eq. 4)
    alpha_n2p_test   : α_{n→p} used at test time (paper uses 0.1, Sec 4.1)
    """

    def __init__(
        self,
        hidden_dim:        int   = 512,
        num_classes:       int   = 7,
        beta_p2n:          float = 0.5,
        beta_v2n:          float = 0.5,
        alpha_p2p:         float = 0.06,
        alpha_v2p:         float = 0.24,
        alpha_n2p_train:   float = 0.2,
        alpha_n2p_test:    float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes      = num_classes
        self.beta_p2n         = beta_p2n
        self.beta_v2n         = beta_v2n
        self.alpha_p2p        = alpha_p2p
        self.alpha_v2p        = alpha_v2p
        self.alpha_n2p_train  = alpha_n2p_train
        self.alpha_n2p_test   = alpha_n2p_test

        # One GCN weight matrix per node type (paper Eq. 1/3/5)
        self.gcn_neg_agg = _GraphModel(hidden_dim)   # W^n
        self.gcn_pos_agg = _GraphModel(hidden_dim)   # W^p
        self.gcn_vis_agg = _GraphModel(hidden_dim)   # W^v

    def forward(
        self,
        text_adapted: torch.Tensor,   # [C, D]  positive text (post-JointAdapter)
        neg_adapted:  torch.Tensor,   # [C, D]  negative text (post-JointAdapter)
        vis_adapted:  torch.Tensor,   # [C, D]  visual class-means (post-JointAdapter)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        ───────
        text_refined : [C, D]  refined positive text prototypes for classification
        vis_bias     : [C, D]  per-class visual bias to be added to the cache (Eq. 6)
        """
        C = self.num_classes

        # Detached copies for adjacency computation (no grad through structure)
        td = text_adapted.detach().unsqueeze(0)  # [1, C, D]
        nd = neg_adapted.detach().unsqueeze(0)
        vd = vis_adapted.detach().unsqueeze(0)

        # ── Pre-compute all adjacency matrices (from detached adapted features) ──
        # Faithful to HeGraph: all adjacencies are fixed before any update step.
        pos_neg_adj = _compute_adj(nd, td, negative=True)    # N←P  [1,C,C]
        vis_neg_adj = _compute_adj(nd, vd, negative=True)    # N←V  [1,C,C]
        pos_pos_adj = _compute_adj(td, td)                   # P←P  [1,C,C]
        pos_pos_adj[0, range(C), range(C)] = 0.0             # remove self-loops
        vis_pos_adj = _compute_adj(td, vd)                   # P←V  [1,C,C]
        neg_pos_adj = _compute_adj(td, nd, negative=True)    # P←N  [1,C,C] from BASE neg
        vis_vis_adj = _compute_adj(vd, vd)                   # V←V  [1,C,C]
        vis_vis_adj[0, range(C), range(C)] = 0.0             # remove self-loops

        # Non-detached versions for actual message passing (gradients flow here)
        text_h = text_adapted.unsqueeze(0)  # [1, C, D]
        neg_h  = neg_adapted.unsqueeze(0)
        vis_h  = vis_adapted.unsqueeze(0)

        # ── Step 1: Update N (Eq. 1-2) ──────────────────────────────────────────
        pos_neg_out = self.gcn_neg_agg(neg_h, text_h, pos_neg_adj, self_add=True)
        vis_neg_out = self.gcn_neg_agg(neg_h, vis_h,  vis_neg_adj, self_add=True)
        neg_h = neg_h + self.beta_p2n * pos_neg_out + self.beta_v2n * vis_neg_out

        # ── Step 2: Update P using updated N (Eq. 3-4) ──────────────────────────
        pos_pos_out = self.gcn_pos_agg(text_h, text_h, pos_pos_adj, self_add=True)
        vis_pos_out = self.gcn_pos_agg(text_h, vis_h,  vis_pos_adj, self_add=True)
        neg_pos_out = self.gcn_pos_agg(text_h, neg_h,  neg_pos_adj, self_add=True)

        alpha_n2p = self.alpha_n2p_train if self.training else self.alpha_n2p_test
        text_h = (text_h
                  + self.alpha_p2p * pos_pos_out
                  + self.alpha_v2p * vis_pos_out
                  + alpha_n2p     * neg_pos_out)

        # ── Step 3: Visual bias (Eq. 5-6, no self-loop) ─────────────────────────
        vis_vis_out = self.gcn_vis_agg(vis_h, vis_h, vis_vis_adj, self_add=False)
        vis_bias    = vis_vis_out.squeeze(0)     # [C, D]

        return text_h.squeeze(0), vis_bias       # [C, D], [C, D]


# =============================================================================
# MLPHeGraphAdapterModel – full pipeline
# =============================================================================

class MLPHeGraphAdapterModel(nn.Module):
    """
    Full MLPHeGraphAdapter model.

    Stage 1  JointAdapter (shared, same weights for all feature types)
        Applies to: query [B,1536], pos-text [C,1536], neg-text [C,1536],
                    visual class-means [C,1536], per-sample cache [CK,1536]

    Stage 2  HeGraphLearnerMFM
        Heterogeneous GCN refines P and computes V-bias in 512-dim space.

    Stage 3  Dual classification
        Text:   τ · query_norm @ P_norm.T                           [B, C]
        Cache:  τ · exp(sim(query_norm, cache_norm) − 1) @ L       [B, C]
        Test:   combined softmax  (Eq. 8 + prediction fusion)

    Buffers (registered externally by the trainer after construction):
        raw_text_cat  : [C,  3·feat_dim]  concatenated positive text features
        raw_neg_cat   : [C,  3·feat_dim]  concatenated negative text features
        raw_vis_cat   : [C,  3·feat_dim]  concatenated visual class-mean features
        raw_cache_cat : [CK, 3·feat_dim]  concatenated per-sample cache features
        labels_onehot : [CK, C]           one-hot labels for cache samples

    Parameters
    ──────────
    num_fms          : number of foundation models (3)
    feat_dim         : per-FM feature dimension (512)
    hidden_dim       : JointAdapter intermediate width (512)
    common_dim       : JointAdapter output and GCN node dimension (512)
    num_classes      : C
    beta_p2n/v2n     : HeGraph N-update weights
    alpha_p2p/v2p    : HeGraph P-update weights
    alpha_n2p_train/test : HeGraph N→P weight (differs at train vs test)
    gamma            : visual bias scale (Eq. 6)
    cache_output     : weight w of visual logits in combined test prediction
    dropout          : JointAdapter dropout
    logit_scale_init : initial log-temperature; exp(2.3) ≈ 10
    """

    def __init__(
        self,
        num_fms:           int   = 3,
        feat_dim:          int   = 512,
        hidden_dim:        int   = 512,
        common_dim:        int   = 512,
        num_classes:       int   = 7,
        beta_p2n:          float = 0.5,
        beta_v2n:          float = 0.5,
        alpha_p2p:         float = 0.06,
        alpha_v2p:         float = 0.24,
        alpha_n2p_train:   float = 0.2,
        alpha_n2p_test:    float = 0.1,
        gamma:             float = 0.5,
        cache_output:      float = 1.0,
        dropout:           float = 0.1,
        logit_scale_init:  float = 2.3,
    ) -> None:
        super().__init__()
        in_dim = num_fms * feat_dim    # 3 × 512 = 1536

        self.gamma        = gamma
        self.cache_output = cache_output

        self.joint_adapter  = JointAdapter(in_dim, hidden_dim, common_dim, dropout)
        self.hegraph_learner = HeGraphLearnerMFM(
            hidden_dim       = common_dim,
            num_classes      = num_classes,
            beta_p2n         = beta_p2n,
            beta_v2n         = beta_v2n,
            alpha_p2p        = alpha_p2p,
            alpha_v2p        = alpha_v2p,
            alpha_n2p_train  = alpha_n2p_train,
            alpha_n2p_test   = alpha_n2p_test,
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    def forward(
        self,
        query_cat:   torch.Tensor,     # [B, 3·feat_dim]  concatenated multi-FM query
        is_training: bool = False,
    ):
        """
        Training  → returns (logits_text [B,C], logits_cache [B,C])
        Inference → returns combined softmax probabilities [B,C]
        """
        # ── Stage 1: JointAdapter on all features ────────────────────────────
        text_adapted  = self.joint_adapter(self.raw_text_cat)    # [C,  D]
        neg_adapted   = self.joint_adapter(self.raw_neg_cat)     # [C,  D]
        vis_adapted   = self.joint_adapter(self.raw_vis_cat)     # [C,  D]
        cache_adapted = self.joint_adapter(self.raw_cache_cat)   # [CK, D]
        query_adapted = self.joint_adapter(query_cat)            # [B,  D]

        # ── Stage 2: Heterogeneous GCN ───────────────────────────────────────
        text_refined, vis_bias = self.hegraph_learner(
            text_adapted, neg_adapted, vis_adapted
        )  # [C, D], [C, D]

        # ── Apply visual bias to cache (Eq. 6) ───────────────────────────────
        per_sample_bias = self.labels_onehot @ vis_bias            # [CK, D]
        cache_final     = cache_adapted + self.gamma * per_sample_bias  # [CK, D]

        # ── Stage 3: Normalise + dual classification ─────────────────────────
        query_norm = F.normalize(query_adapted, dim=-1)   # [B,  D]
        text_norm  = F.normalize(text_refined,  dim=-1)   # [C,  D]
        cache_norm = F.normalize(cache_final,   dim=-1)   # [CK, D]

        scale = self.logit_scale.exp().clamp(max=100.0)

        # Text-based logits (Eq. 7)
        logits_text = scale * (query_norm @ text_norm.T)           # [B, C]

        # Tip-Adapter visual logits (Eq. 8)
        sim_cache    = query_norm @ cache_norm.T                   # [B, CK]
        A            = torch.exp(sim_cache - 1.0) @ self.labels_onehot  # [B, C]
        logits_cache = scale * A                                   # [B, C]

        if is_training:
            return logits_text, logits_cache

        # Combined prediction: weighted average of softmax outputs
        return (F.softmax(logits_text,  dim=-1)
                + self.cache_output * F.softmax(logits_cache, dim=-1)) / (1.0 + self.cache_output)
