"""
MLPGA Model – Multi-FM Early Fusion + Faithful GraphAdapter Prototype Refinement.

Architecture summary
────────────────────
Stage 1  Early Fusion + JointAdapter
    Concatenate image features from all foundation models:
        [B, 512] × 3  →  cat  →  [B, 1536]
    Project to common embedding space via MLP:
        JointAdapter: [B, 1536]  →  [B, common_dim]
    The same adapter is applied to visual and text prototypes so that
    queries and prototypes live in the same metric space.

Stage 2  Faithful GraphAdapter Text Prototype Refinement
    GCN_tt  – Text-to-text prototype graph
        All C class text prototypes form a fully-connected (then sparsified) graph.
        Each text node attends to all other text prototypes via cosine-similarity edges.
        Output: refined text prototypes  [C, common_dim]

    GCN_it  – Image-prototype-to-text prototype graph
        For each class c, build a (1 + C)-node graph:
            node 0          : text prototype of class c
            nodes 1..C      : all visual prototypes
        One GCN pass → extract node 0 → refined text_c  [common_dim]
        Stack over all C classes → [C, common_dim]

    Blending (fixed scalars, matching the original paper):
        graph_out  = β * GCN_tt_out     + (1 − β) * GCN_it_out       (β: text-stream weight, z'_t = β·z_tt + (1−β)·z_vt)
        final_text = α * tproto_adapted + (1 − α) * graph_out        (α: original-text weight, z*_t = α·z_t + (1−α)·z'_t)

    Query: passes through JointAdapter only — no graph refinement.
    This faithfully follows the GraphAdapter paper where the query is never
    injected into any graph and is used unchanged at classification time.

Stage 3  Classification
    L2-normalise both query_adapted and final_text.
    logits = exp(logit_scale) * query_adapted @ final_text.T   [B, C]

Design note on α / β
    Both α and β are fixed scalars (not learnable), matching the original
    GraphAdapter paper. This keeps the graph hyperparameters consistent
    with the paper's design and avoids introducing additional learnable
    routing weights in the low-shot regime.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# JointAdapter – maps concatenated multi-FM features to a common space
# =============================================================================

class JointAdapter(nn.Module):
    """
    Two-layer MLP that projects concatenated FM features into a common space.

    Applied identically to queries, visual prototypes, and text prototypes so
    all three live in the same metric space for graph construction and cosine
    classification.

    Parameters
    ──────────
    in_dim     : int   concatenated input dimension (e.g. 3 × 512 = 1536)
    hidden_dim : int   intermediate MLP width (e.g. 512)
    common_dim : int   output embedding dimension (e.g. 512)
    dropout    : float dropout probability applied after the first activation
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        common_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, common_dim),
            nn.LayerNorm(common_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# GraphConvolution – single GCN layer
# =============================================================================

class GraphConvolution(nn.Module):
    """
    One graph convolutional layer with symmetric normalisation.

    Implements:
        H' = Tanh( D^{-½} (A + I) D^{-½}  H  W  +  b )

    where A is the non-negative input adjacency and I adds self-loops.
    The bias b has shape [num_nodes, out_dim] — one vector per node,
    matching the original GraphAdapter convention.

    Parameters
    ──────────
    in_dim    : int  input feature dimension per node
    out_dim   : int  output feature dimension per node
    num_nodes : int  expected number of nodes (fixes bias shape)
    bias      : bool whether to add a per-node bias term
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_nodes: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias_param: nn.Parameter | None = (
            nn.Parameter(torch.zeros(num_nodes, out_dim)) if bias else None
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ──────────
        x   : [N, in_dim]   node feature matrix
        adj : [N, N]        symmetric, non-negative, self-loops NOT included

        Returns
        ───────
        out : [N, out_dim]
        """
        N = adj.size(0)
        adj = adj.clamp(min=0.0)
        adj = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        d = adj.sum(dim=-1).clamp(min=1e-8).pow(-0.5)
        D_inv_sqrt = torch.diag(d)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        h   = x @ self.weight
        out = adj_norm @ h
        if self.bias_param is not None:
            out = out + self.bias_param
        return torch.tanh(out)


# =============================================================================
# GraphLearner – faithful GraphAdapter text prototype refinement
# =============================================================================

class GraphLearner(nn.Module):
    """
    GraphAdapter-faithful prototype refinement in the common embedding space.

    Two GCN streams both refine TEXT prototypes:

        GCN_tt  for each class c, places text_c as node 0 with all C text
                prototypes as context neighbours → text-to-text refinement.

        GCN_it  for each class c, places text_c as node 0 with all C visual
                prototypes as context neighbours → cross-modal refinement.
                After one GCN pass, node 0 carries cross-modal context
                from the visual prototype neighbourhood.

    Blending (fixed weights, matching the GraphAdapter paper):
        graph_out  = β * GCN_tt_out     + (1 − β) * GCN_it_out       (β weights the text stream)
        final_text = α * tproto_adapted + (1 − α) * graph_out        (α weights the original text)

    The query is NOT touched — it passes through JointAdapter only and is
    used unchanged for cosine classification (faithful to the paper).

    Parameters
    ──────────
    dim         : int   node feature dimension (= common_dim)
    num_classes : int   number of class prototypes C
    alpha_it    : float paper's α — weight on the ORIGINAL text feature in the residual (paper optimal 0.6)
    beta_tt     : float paper's β — weight on the text (intra) stream when fusing the two GCNs (paper optimal 0.7)
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        alpha_it: float = 0.6,
        beta_tt:  float = 0.7,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.num_classes = num_classes

        self.alpha = alpha_it
        self.beta  = beta_tt

        # Both GCNs use per-class (1+C)-node graphs: node 0 = text_c, nodes 1..C = context.
        # GCN_tt context: all text protos.  GCN_it context: all visual protos.
        self.gcn_tt = GraphConvolution(dim, dim, num_nodes=1 + num_classes)
        self.gcn_it = GraphConvolution(dim, dim, num_nodes=1 + num_classes)

    def _build_adj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dense, non-negative cosine-similarity adjacency (all positive edges kept,
        matching the original GraphAdapter design — no top-k sparsification).
        Adjacency is detached — no gradient flows through edge weights.
        """
        x_norm = F.normalize(x.detach(), dim=-1)
        sim    = x_norm @ x_norm.T
        N      = sim.size(0)

        sim.fill_diagonal_(0.0)
        sim = sim.clamp(min=0.0)

        return sim

    def forward(
        self,
        vproto_adapted: torch.Tensor,   # [C, dim]  visual protos in common space
        tproto_adapted: torch.Tensor,   # [C, dim]  text protos in common space
    ) -> torch.Tensor:
        """
        Returns
        ───────
        final_text : [C, dim]  refined text prototypes for classification
        """
        C = self.num_classes

        # Both GCN streams use the same per-class graph structure (faithful to paper):
        #   node 0   : text prototype of class c  (the node being refined)
        #   nodes 1..C : context nodes (text protos for GCN_tt, visual protos for GCN_it)
        # After one GCN pass, extract node 0 → refined text_c.

        # ── GCN_tt: text_c (node 0) + all text protos as context ─────────
        gcn_tt_list: List[torch.Tensor] = []
        for c in range(C):
            t_c    = tproto_adapted[c:c+1]                         # [1, dim]
            nodes  = torch.cat([t_c, tproto_adapted], dim=0)       # [1+C, dim]
            adj_tt = self._build_adj(nodes)                        # [1+C, 1+C]
            out_tt = self.gcn_tt(nodes, adj_tt)                    # [1+C, dim]
            gcn_tt_list.append(out_tt[0])                          # [dim]
        gcn_tt_out = torch.stack(gcn_tt_list, dim=0)               # [C, dim]

        # ── GCN_it: text_c (node 0) + all visual protos as context ───────
        gcn_it_list: List[torch.Tensor] = []
        for c in range(C):
            t_c    = tproto_adapted[c:c+1]                         # [1, dim]
            nodes  = torch.cat([t_c, vproto_adapted], dim=0)       # [1+C, dim]
            adj_it = self._build_adj(nodes)                        # [1+C, 1+C]
            out_it = self.gcn_it(nodes, adj_it)                    # [1+C, dim]
            gcn_it_list.append(out_it[0])                          # [dim]
        gcn_it_out = torch.stack(gcn_it_list, dim=0)               # [C, dim]

        # ── Blend the two GCN streams: β (beta) weights the text stream ───
        #    Paper Eq.:  z'_t = β·z_tt + (1−β)·z_vt      (paper optimal β = 0.7)
        graph_out  = self.beta * gcn_tt_out + (1.0 - self.beta) * gcn_it_out   # [C, dim]

        # ── Residual: α (alpha) weights the ORIGINAL text feature ─────────
        #    Paper Eq.:  z*_t = α·z_t + (1−α)·z'_t       (paper optimal α = 0.6)
        final_text = self.alpha * tproto_adapted + (1.0 - self.alpha) * graph_out  # [C, dim]

        return final_text


# =============================================================================
# MLPGAModel – full pipeline
# =============================================================================

class MLPGAModel(nn.Module):
    """
    Full MLPGA model (all trainable components).

    Stage 1  JointAdapter
        query_feats:   list[num_fms] of [B, feat_dim]  →  cat  →  [B, num_fms*feat_dim]
                                                        → adapter →  [B, common_dim]
        vproto_feats:  list[num_fms] of [C, feat_dim]  →  (same) →  [C, common_dim]
        tproto_feats:  list[num_fms] of [C, feat_dim]  →  (same) →  [C, common_dim]

    Stage 2  GraphLearner (text-only refinement)
        GCN_tt + GCN_it → blended → residual
        final_text  [C, common_dim]
        query_adapted is returned unchanged from Stage 1.

    Stage 3  Classification
        L2-normalise + cosine: logits = exp(s) * query_adapted @ final_text.T  [B, C]

    Parameters
    ──────────
    num_fms          : int   number of foundation models (e.g. 3)
    feat_dim         : int   output dimension of each FM (all 512 in this project)
    hidden_dim       : int   intermediate width of JointAdapter (e.g. 512)
    common_dim       : int   output dimension of JointAdapter / GCN width (e.g. 512)
    num_classes      : int   number of target classes C
    alpha_it         : float paper's α — original-text residual weight (paper optimal 0.6)
    beta_tt          : float paper's β — text-stream fusion weight (paper optimal 0.7)
    dropout          : float dropout in JointAdapter
    logit_scale_init : float initial log-temperature for cosine classifier
    """

    def __init__(
        self,
        num_fms:          int,
        feat_dim:         int,
        hidden_dim:       int,
        common_dim:       int,
        num_classes:      int,
        alpha_it:         float = 0.6,
        beta_tt:          float = 0.7,
        dropout:          float = 0.1,
        logit_scale_init: float = 2.3,
    ) -> None:
        super().__init__()
        in_dim = num_fms * feat_dim

        self.joint_adapter = JointAdapter(in_dim, hidden_dim, common_dim, dropout)
        self.graph_learner  = GraphLearner(
            dim         = common_dim,
            num_classes = num_classes,
            alpha_it    = alpha_it,
            beta_tt     = beta_tt,
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    def forward(
        self,
        query_features:  List[torch.Tensor],   # list[num_fms] of [B, feat_dim]
        vproto_features: List[torch.Tensor],   # list[num_fms] of [C, feat_dim]
        tproto_features: List[torch.Tensor],   # list[num_fms] of [C, feat_dim]
    ) -> torch.Tensor:
        """Returns logits [B, C]."""
        # ── Stage 1: early fusion + joint adaptation ───────────────────────
        query_cat   = torch.cat(query_features,  dim=-1)   # [B, num_fms*feat_dim]
        vproto_cat  = torch.cat(vproto_features, dim=-1)   # [C, num_fms*feat_dim]
        tproto_cat  = torch.cat(tproto_features, dim=-1)   # [C, num_fms*feat_dim]

        query_adapted  = self.joint_adapter(query_cat)     # [B, common_dim]
        vproto_adapted = self.joint_adapter(vproto_cat)    # [C, common_dim]
        tproto_adapted = self.joint_adapter(tproto_cat)    # [C, common_dim]

        # ── Stage 2: GraphAdapter-faithful text prototype refinement ───────
        # Query is NOT passed to the graph — it is used unchanged (see paper).
        final_text = self.graph_learner(vproto_adapted, tproto_adapted)  # [C, common_dim]

        # ── Stage 3: cosine classification ────────────────────────────────
        final_query = F.normalize(query_adapted, dim=-1)   # [B, common_dim]
        final_text  = F.normalize(final_text,    dim=-1)   # [C, common_dim]

        scale  = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (final_query @ final_text.T)      # [B, C]
        return logits
