"""
Training script for Multi-Foundation Model Fusion Framework

This script provides the entry point for training models using the DASSL framework.
It supports both the MultiFMFusion trainer (main contribution) and baseline trainers.

Example usage:
    # Train MultiFMFusion model
    python train.py \
        --root /path/to/datasets \
        --output-dir output/multi_fm_bracs \
        --trainer MultiFMFusion \
        --dataset-config-file configs/datasets/bracs_descriptive.yaml \
        --config-file configs/trainers/multi_fm_fusion.yaml \
        DATASET.NUM_SHOTS 16 \
        SEED 1

    # Evaluation only
    python train.py \
        --root /path/to/datasets \
        --trainer MultiFMFusion \
        --dataset-config-file configs/datasets/bracs_descriptive.yaml \
        --config-file configs/trainers/multi_fm_fusion.yaml \
        --eval-only \
        --model-dir output/multi_fm_bracs
"""

import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


# =============================================================================
# Import custom datasets (auto-registers with DASSL)
# =============================================================================

# Pathology datasets (descriptive class names version)
import datasets.lungHist700_tien_descriptive
import datasets.bracs_tien_descriptive
import datasets.iciar2018_tien_descriptive


# =============================================================================
# Import custom trainers (auto-registers with DASSL)
# =============================================================================

import trainers.mlpga_trainer      # MLPGAFaithful
import trainers.ctfig_mvc_trainer       # CrossFMTransductiveInstanceGraphMVC
import trainers.mlphegraph_trainer      # MLPHeGraphAdapter

# Model registry
import trainers.model_registry


# =============================================================================
# Config Utilities
# =============================================================================

def print_args(args, cfg):
    """Print arguments and config for logging."""
    print("="*60)
    print("Arguments")
    print("="*60)
    for key in sorted(vars(args)):
        print(f"  {key}: {getattr(args, key)}")
    print("="*60)
    print("Config")
    print("="*60)
    print(cfg)


def reset_cfg(cfg, args):
    """Override config with CLI arguments."""
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.init_weights:
        cfg.MODEL.INIT_WEIGHTS = args.init_weights


def extend_cfg(cfg):
    """
    Add custom config variables for the multi-FM fusion framework.
    """
    from yacs.config import CfgNode as CN

    # =================================================================
    # Multi-FM Fusion specific configs
    # =================================================================
    cfg.TRAINER.MULTIFM = CN()
    
    # Foundation models to use
    cfg.TRAINER.MULTIFM.FM_NAMES = ["biomedclip", "plip", "conch"]
    
    # Common embedding dimension (all FMs mapped to this)
    cfg.TRAINER.MULTIFM.COMMON_DIM = 512
    
    # Fusion type: "late" (attention-weighted) or "hierarchical" (meta-graph GCN)
    cfg.TRAINER.MULTIFM.FUSION_TYPE = "late"
    
    # Number of GCN layers per graph
    cfg.TRAINER.MULTIFM.GCN_LAYERS = 2
    
    # Cross-modal edge weight initialization
    cfg.TRAINER.MULTIFM.CROSS_MODAL_INIT = 0.5
    
    # Adapter config
    cfg.TRAINER.MULTIFM.ADAPTER_HIDDEN = None  # None for linear, int for MLP

    # Contrastive loss
    cfg.TRAINER.MULTIFM.CONTRASTIVE_WEIGHT = 0.1
    cfg.TRAINER.MULTIFM.CONTRASTIVE_TEMP = 0.07

    # Graph variant: "gcn" (legacy) or "gat" (per-FM, multi-head)
    cfg.TRAINER.MULTIFM.GRAPH_TYPE = "gcn"
    # Number of attention heads for GAT
    cfg.TRAINER.MULTIFM.GAT_HEADS = 4
    # Top-K sparsification of adjacency (0 = dense; e.g., 3 keeps top-3 neighbors)
    cfg.TRAINER.MULTIFM.GRAPH_TOP_K = 0
    # Initial logit scale (log temperature). exp(2.3) ~= 10
    cfg.TRAINER.MULTIFM.LOGIT_SCALE_INIT = 2.3
    # Optional path to a precomputed test-feature cache (.pt produced by
    # precompute_test_features.py). When set, the trainer skips the FM
    # forward pass during evaluation and looks features up by impath.
    cfg.TRAINER.MULTIFM.TEST_CACHE_PATH = ""

    # =================================================================
    # Multi-FM GraphAdapter (per-FM dual class graph) configs
    # =================================================================
    cfg.TRAINER.MULTIFM_GA = CN()
    cfg.TRAINER.MULTIFM_GA.FM_NAMES = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.MULTIFM_GA.ALPHA = 0.6                # residual weight on original text feature
    cfg.TRAINER.MULTIFM_GA.BETA = 0.7                 # text-graph vs visual-graph weight
    cfg.TRAINER.MULTIFM_GA.DROPOUT = 0.1
    cfg.TRAINER.MULTIFM_GA.LOGIT_SCALE_INIT = 2.3
    cfg.TRAINER.MULTIFM_GA.LEARNABLE_FM_WEIGHTS = False
    cfg.TRAINER.MULTIFM_GA.LEARNABLE_ALPHABETA = False
    cfg.TRAINER.MULTIFM_GA.CONTRASTIVE_WEIGHT = 0.0   # off by default; reuse SupConLoss if >0
    cfg.TRAINER.MULTIFM_GA.CONTRASTIVE_TEMP = 0.07
    # Image-side adapter (CLIP-Adapter style, per FM)
    cfg.TRAINER.MULTIFM_GA.USE_IMAGE_ADAPTER = False
    cfg.TRAINER.MULTIFM_GA.IMG_ADAPTER_BOTTLENECK = 128
    cfg.TRAINER.MULTIFM_GA.IMG_ADAPTER_GAMMA_INIT = 0.2

    # =================================================================
    # HetXFM (heterogeneous cross-FM multi-view graph)
    # =================================================================
    cfg.TRAINER.HETXFM = CN()
    cfg.TRAINER.HETXFM.FM_NAMES = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.HETXFM.COMMON_DIM = 256
    cfg.TRAINER.HETXFM.R_GAT_LAYERS = 2
    cfg.TRAINER.HETXFM.R_GAT_HEADS = 4
    cfg.TRAINER.HETXFM.R_GAT_DROPOUT = 0.1
    cfg.TRAINER.HETXFM.USE_HYPEREDGE = True
    cfg.TRAINER.HETXFM.TOP_K = 3
    cfg.TRAINER.HETXFM.DROP_EDGE = 0.3
    cfg.TRAINER.HETXFM.EDGE_DROPOUT = 0.2
    cfg.TRAINER.HETXFM.GATE_INIT_BIAS = [0.0, 0.0, 1.0]
    cfg.TRAINER.HETXFM.AGREEMENT_WEIGHT = 0.05
    cfg.TRAINER.HETXFM.ALIGN_WEIGHT = 0.05
    cfg.TRAINER.HETXFM.ALIGN_TEMP = 0.07
    cfg.TRAINER.HETXFM.LABEL_SMOOTHING = 0.1
    cfg.TRAINER.HETXFM.STAGE1_EPOCHS = 5
    cfg.TRAINER.HETXFM.STAGE1_LOSS = "align_only"
    cfg.TRAINER.HETXFM.PROTO_REFRESH_EVERY = 5
    cfg.TRAINER.HETXFM.LOGIT_SCALE_INIT = 2.3
    cfg.TRAINER.HETXFM.STOCHASTIC_DEPTH = 0.1

    # =================================================================
    # Legacy configs (for compatibility with existing trainers)
    # =================================================================
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

    # Graph adapter tuning defaults
    cfg.TRAINER.GRAPHADAPTER = CN()
    cfg.TRAINER.GRAPHADAPTER.BETA = 0.7
    cfg.TRAINER.GRAPHADAPTER.ALPHA = 0.6

    # HeGraph specific (for baseline comparisons)
    cfg.TRAINER.HEGRAPH = CN()
    cfg.TRAINER.HEGRAPH.CACHE_RATIO = 0.1
    cfg.TRAINER.HEGRAPH.CACHE_OUTPUT = 1.0
    cfg.TRAINER.HEGRAPH.BETA_P2N = 0.5
    cfg.TRAINER.HEGRAPH.BETA_V2N = 0.5
    cfg.TRAINER.HEGRAPH.ALPHA_P2P = 0.06
    cfg.TRAINER.HEGRAPH.ALPHA_V2P = 0.24
    cfg.TRAINER.HEGRAPH.ALPHA_N2P_TRAIN = 0.2
    cfg.TRAINER.HEGRAPH.ALPHA_N2P_TEST = 0.1
    cfg.TRAINER.HEGRAPH.GAMMA = 0.5
    cfg.TRAINER.HEGRAPH.MLP_DROPOUT = 0.1

    # =================================================================
    # MVGraph (multi-view prototype graph)
    # =================================================================
    cfg.TRAINER.MVGRAPH = CN()
    cfg.TRAINER.MVGRAPH.FM_NAMES = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.MVGRAPH.COMMON_DIM = 256       # shared embedding dimension
    cfg.TRAINER.MVGRAPH.GNN_TYPE = "gat"       # "gat" or "gcn"
    cfg.TRAINER.MVGRAPH.NUM_GNN_LAYERS = 1     # layers per per-view GNN
    cfg.TRAINER.MVGRAPH.GNN_HEADS = 4          # attention heads (GAT only)
    cfg.TRAINER.MVGRAPH.TOP_K = 3              # top-k neighbours per prototype node
    cfg.TRAINER.MVGRAPH.DROPOUT = 0.1
    cfg.TRAINER.MVGRAPH.CV_LOSS_WEIGHT = 0.3   # cross-view InfoNCE weight
    cfg.TRAINER.MVGRAPH.USE_TEXT_ALIGN = True  # text-visual alignment loss
    cfg.TRAINER.MVGRAPH.ALIGN_WEIGHT = 0.1
    cfg.TRAINER.MVGRAPH.LOGIT_SCALE_INIT = 2.3
    cfg.TRAINER.MVGRAPH.PREC = "fp32"

    # =================================================================
    # MVFuse (multi-view early fusion graph)
    # =================================================================
    cfg.TRAINER.MVFUSE = CN()
    cfg.TRAINER.MVFUSE.FM_NAMES = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.MVFUSE.COMMON_DIM = 256        # shared embedding dimension
    cfg.TRAINER.MVFUSE.NUM_VIEWS = 3           # K multi-view projection heads
    cfg.TRAINER.MVFUSE.GNN_TYPE = "gat"        # "gat" or "gcn"
    cfg.TRAINER.MVFUSE.NUM_GNN_LAYERS = 1      # layers per per-view GNN
    cfg.TRAINER.MVFUSE.GNN_HEADS = 4           # attention heads (GAT only)
    cfg.TRAINER.MVFUSE.TOP_K = 3               # top-k neighbours per prototype node
    cfg.TRAINER.MVFUSE.DROPOUT = 0.1
    cfg.TRAINER.MVFUSE.CV_LOSS_WEIGHT = 0.1    # cross-view InfoNCE weight
    cfg.TRAINER.MVFUSE.USE_TEXT_ALIGN = True   # text-visual alignment loss
    cfg.TRAINER.MVFUSE.ALIGN_WEIGHT = 0.05
    cfg.TRAINER.MVFUSE.LOGIT_SCALE_INIT = 2.3
    cfg.TRAINER.MVFUSE.LABEL_SMOOTHING = 0.1   # CE label smoothing
    cfg.TRAINER.MVFUSE.PREC = "fp32"

    # =================================================================
    # MVQPGraph (multi-view query-prototype graph)
    # =================================================================
    cfg.TRAINER.MVQPGRAPH = CN()
    cfg.TRAINER.MVQPGRAPH.FM_NAMES = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.MVQPGRAPH.COMMON_DIM = 256
    cfg.TRAINER.MVQPGRAPH.GNN_TYPE = "gat"
    cfg.TRAINER.MVQPGRAPH.NUM_GNN_LAYERS = 1
    cfg.TRAINER.MVQPGRAPH.GNN_HEADS = 4
    cfg.TRAINER.MVQPGRAPH.TOP_K = 0              # 0 = fully connected (recommended)
    cfg.TRAINER.MVQPGRAPH.DROPOUT = 0.1
    cfg.TRAINER.MVQPGRAPH.CV_LOSS_WEIGHT = 0.1
    cfg.TRAINER.MVQPGRAPH.LABEL_SMOOTHING = 0.1
    cfg.TRAINER.MVQPGRAPH.LOGIT_SCALE_INIT = 2.3
    cfg.TRAINER.MVQPGRAPH.PREC = "fp32"
    cfg.TRAINER.MVQPGRAPH.USE_GNN = True          # False = adapter-only ablation

    # =================================================================
    # MLPGraphAdapter (early MLP fusion + GraphAdapter GCN refinement)
    # =================================================================
    cfg.TRAINER.MLPGA = CN()
    cfg.TRAINER.MLPGA.FM_NAMES         = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.MLPGA.COMMON_DIM       = 512   # GCN and output dimension
    cfg.TRAINER.MLPGA.HIDDEN_DIM       = 512   # JointAdapter intermediate width
    cfg.TRAINER.MLPGA.ALPHA_IT         = 0.7   # initial GCN_tt blend weight
    cfg.TRAINER.MLPGA.BETA_IT          = 0.5   # initial GCN_it blend weight (MLPGraphAdapter)
    cfg.TRAINER.MLPGA.BETA_TT          = 0.7   # initial text-residual blend weight (MLPGAFaithful)
    cfg.TRAINER.MLPGA.DROPOUT          = 0.1
    cfg.TRAINER.MLPGA.LOGIT_SCALE_INIT = 2.3
    cfg.TRAINER.MLPGA.PREC             = "fp32"
    # Optional path to precomputed test-feature cache (.pt from
    # precompute_test_features.py). Used only by MLPGraphAdapterCached.
    cfg.TRAINER.MLPGA.TEST_CACHE_PATH  = ""

    # =================================================================
    # DualGraph (per-FM adapters + shared GAT + attention readout + MLP)
    # =================================================================
    cfg.TRAINER.DUALGRAPH = CN()
    cfg.TRAINER.DUALGRAPH.FM_NAMES          = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.DUALGRAPH.COMMON_DIM        = 512   # adapter output = shared space dim
    cfg.TRAINER.DUALGRAPH.HIDDEN_DIM        = 512   # FMAdapter MLP hidden width
    cfg.TRAINER.DUALGRAPH.GAT_LAYERS        = 1     # shared GAT depth
    cfg.TRAINER.DUALGRAPH.GAT_HEADS         = 4     # attention heads
    cfg.TRAINER.DUALGRAPH.GAT_DROPOUT       = 0.1
    cfg.TRAINER.DUALGRAPH.CLASSIFIER_HIDDEN = 512   # MLP classifier hidden dim
    cfg.TRAINER.DUALGRAPH.DROPOUT           = 0.1   # dropout in adapters and classifier
    cfg.TRAINER.DUALGRAPH.LABEL_SMOOTHING   = 0.1
    cfg.TRAINER.DUALGRAPH.PREC              = "fp32"

    # =================================================================
    # CrossFMProtoGraph (cross-FM prototype graph, no early MLP fusion)
    # =================================================================
    cfg.TRAINER.CFPG = CN()
    cfg.TRAINER.CFPG.FM_NAMES           = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.CFPG.NUM_GNN_LAYERS     = 1      # CFPG depth; 0 = per-FM ProtoNet + late fusion
    cfg.TRAINER.CFPG.TOP_K              = 3      # top-k neighbours in intra-FM adjacency (0=dense)
    cfg.TRAINER.CFPG.DROPOUT            = 0.1
    cfg.TRAINER.CFPG.GATE_CV_INIT       = 0.0   # raw init for intra-FM T->V gate (sigmoid(0)=0.5)
    cfg.TRAINER.CFPG.GATE_CT_INIT       = 0.0   # raw init for intra-FM V->T gate
    cfg.TRAINER.CFPG.GATE_XV_INIT       = 0.0   # raw init for cross-FM visual gate
    cfg.TRAINER.CFPG.GATE_XT_INIT       = 0.0   # raw init for cross-FM text gate
    cfg.TRAINER.CFPG.ALPHA_V_INIT       = 0.5   # initial sigma(alpha_v): visual residual blend
    cfg.TRAINER.CFPG.ALPHA_T_INIT       = 0.5   # initial sigma(alpha_t): text   residual blend
    cfg.TRAINER.CFPG.LOGIT_SCALE_INIT   = 2.3   # log-temperature; exp(2.3) approx 10
    cfg.TRAINER.CFPG.LABEL_SMOOTHING    = 0.1
    cfg.TRAINER.CFPG.ALIGN_LOSS_WEIGHT  = 0.0   # weight for optional per-FM V-T alignment loss
    cfg.TRAINER.CFPG.USE_INTRAMODAL_GNN = True  # enable V->V / T->T intra-FM edges
    cfg.TRAINER.CFPG.USE_CROSSMODAL_GNN = True  # enable T->V / V->T intra-FM edges
    cfg.TRAINER.CFPG.USE_CROSSFM_GNN    = True  # enable cross-FM same-class edges (key novelty)
    cfg.TRAINER.CFPG.CLASSIFY_WITH      = "text"  # "text" | "visual" | "average"
    cfg.TRAINER.CFPG.PREC               = "fp32"

    # =================================================================
    # CrossFMInstanceGraph (instance-level cross-FM graph)
    # =================================================================
    cfg.TRAINER.CFIG = CN()
    cfg.TRAINER.CFIG.FM_NAMES           = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.CFIG.NUM_GNN_LAYERS     = 1
    cfg.TRAINER.CFIG.TOP_K              = 3
    cfg.TRAINER.CFIG.DROPOUT            = 0.1
    cfg.TRAINER.CFIG.GATE_CV_INIT       = 0.0
    cfg.TRAINER.CFIG.GATE_CT_INIT       = 0.0
    cfg.TRAINER.CFIG.GATE_XV_INIT       = 0.0
    cfg.TRAINER.CFIG.GATE_XT_INIT       = 0.0
    cfg.TRAINER.CFIG.ALPHA_V_INIT       = 0.5
    cfg.TRAINER.CFIG.ALPHA_T_INIT       = 0.5
    cfg.TRAINER.CFIG.LOGIT_SCALE_INIT   = 2.3
    cfg.TRAINER.CFIG.LABEL_SMOOTHING    = 0.1
    cfg.TRAINER.CFIG.ALIGN_LOSS_WEIGHT  = 0.0
    cfg.TRAINER.CFIG.USE_INTRAMODAL_GNN = True
    cfg.TRAINER.CFIG.USE_CROSSMODAL_GNN = True
    cfg.TRAINER.CFIG.USE_CROSSFM_GNN    = True  # same-instance cross-FM edges (key novelty)
    cfg.TRAINER.CFIG.CLASSIFY_WITH      = "text"
    cfg.TRAINER.CFIG.PREC               = "fp32"

    # =================================================================
    # CrossFMInstanceGraphAttn (instance graph + attention pooling)
    # =================================================================
    cfg.TRAINER.CFIGA = CN()
    cfg.TRAINER.CFIGA.FM_NAMES           = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.CFIGA.NUM_GNN_LAYERS     = 1
    cfg.TRAINER.CFIGA.TOP_K              = 3
    cfg.TRAINER.CFIGA.DROPOUT            = 0.1
    cfg.TRAINER.CFIGA.GATE_CV_INIT       = 0.0
    cfg.TRAINER.CFIGA.GATE_CT_INIT       = 0.0
    cfg.TRAINER.CFIGA.GATE_XV_INIT       = 0.0
    cfg.TRAINER.CFIGA.GATE_XT_INIT       = 0.0
    cfg.TRAINER.CFIGA.ALPHA_V_INIT       = 0.5
    cfg.TRAINER.CFIGA.ALPHA_T_INIT       = 0.5
    cfg.TRAINER.CFIGA.LOGIT_SCALE_INIT   = 2.3
    cfg.TRAINER.CFIGA.LABEL_SMOOTHING    = 0.1
    cfg.TRAINER.CFIGA.ALIGN_LOSS_WEIGHT  = 0.0
    cfg.TRAINER.CFIGA.USE_INTRAMODAL_GNN = True
    cfg.TRAINER.CFIGA.USE_CROSSMODAL_GNN = True
    cfg.TRAINER.CFIGA.USE_CROSSFM_GNN    = True
    cfg.TRAINER.CFIGA.CLASSIFY_WITH      = "text"
    cfg.TRAINER.CFIGA.PREC               = "fp32"

    # =================================================================
    # CrossFMTransductiveInstanceGraph (transductive query nodes + MLP)
    # =================================================================
    cfg.TRAINER.CTFIG = CN()
    cfg.TRAINER.CTFIG.FM_NAMES           = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.CTFIG.NUM_GNN_LAYERS     = 1
    cfg.TRAINER.CTFIG.TOP_K              = 3
    cfg.TRAINER.CTFIG.DROPOUT            = 0.1
    cfg.TRAINER.CTFIG.MLP_HIDDEN_DIM     = 768
    cfg.TRAINER.CTFIG.GATE_CV_INIT       = 0.0
    cfg.TRAINER.CTFIG.GATE_CT_INIT       = 0.0
    cfg.TRAINER.CTFIG.GATE_XV_INIT       = 0.0
    cfg.TRAINER.CTFIG.GATE_XT_INIT       = 0.0
    cfg.TRAINER.CTFIG.ALPHA_V_INIT       = 0.5
    cfg.TRAINER.CTFIG.ALPHA_T_INIT       = 0.5
    cfg.TRAINER.CTFIG.GATE_VT_INIT       = 0.0
    cfg.TRAINER.CTFIG.LOGIT_SCALE_INIT   = 2.3
    cfg.TRAINER.CTFIG.LABEL_SMOOTHING    = 0.1
    cfg.TRAINER.CTFIG.USE_INTRAMODAL_GNN = True
    cfg.TRAINER.CTFIG.USE_CROSSMODAL_GNN = True
    cfg.TRAINER.CTFIG.USE_CROSSFM_GNN    = True
    cfg.TRAINER.CTFIG.USE_TRANSDUCTIVE   = True
    cfg.TRAINER.CTFIG.USE_PROTOTYPE_NODES = False
    cfg.TRAINER.CTFIG.PREC               = "fp32"
    cfg.TRAINER.CTFIG.INIT_WEIGHTS       = ""
    cfg.TRAINER.CTFIG.SHARE_CROSSMODAL_W = False
    cfg.TRAINER.CTFIG.CROSSFM_ATTENTION  = False
    cfg.TRAINER.CTFIG.LEARNED_FM_WEIGHTS = False

    # =================================================================
    # CrossModalProtoGraph (heterogeneous prototype graph)
    # =================================================================
    cfg.TRAINER.CMPG = CN()
    cfg.TRAINER.CMPG.FM_NAMES           = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.CMPG.COMMON_DIM         = 512    # JointAdapter output width D; GCN node dim
    cfg.TRAINER.CMPG.HIDDEN_DIM         = 512    # JointAdapter intermediate MLP width
    cfg.TRAINER.CMPG.NUM_GNN_LAYERS     = 1      # HeteroProtoGCN depth; 0 = MLP baseline
    cfg.TRAINER.CMPG.TOP_K              = 3      # top-k neighbours in adjacency (0 = dense)
    cfg.TRAINER.CMPG.DROPOUT            = 0.1
    cfg.TRAINER.CMPG.ALPHA_V_INIT       = 0.5   # initial σ(α_v): visual residual blend weight
    cfg.TRAINER.CMPG.ALPHA_T_INIT       = 0.5   # initial σ(α_t): text   residual blend weight
    cfg.TRAINER.CMPG.GATE_V_INIT        = 0.0   # raw init for V-node cross-modal gate (σ(0)=0.5)
    cfg.TRAINER.CMPG.GATE_T_INIT        = 0.0   # raw init for T-node cross-modal gate (σ(0)=0.5)
    cfg.TRAINER.CMPG.LOGIT_SCALE_INIT   = 2.3   # log-temperature; exp(2.3) ≈ 10
    cfg.TRAINER.CMPG.LABEL_SMOOTHING    = 0.1
    cfg.TRAINER.CMPG.ALIGN_LOSS_WEIGHT  = 0.0   # weight for optional V-T alignment loss (0=off)
    cfg.TRAINER.CMPG.USE_INTRAMODAL_GNN = True  # enable V-V / T-T intra-modal edges
    cfg.TRAINER.CMPG.USE_CROSSMODAL_GNN = True  # enable V←T / T←V cross-modal edges
    cfg.TRAINER.CMPG.CLASSIFY_WITH      = "text"  # "text" | "visual" | "average"
    cfg.TRAINER.CMPG.PREC               = "fp32"

    # General model configs
    # =================================================================
    # MLPHeGraphAdapter (early MLP fusion + HeGraph heterogeneous GCN)
    # =================================================================
    cfg.TRAINER.MLPHEGRAPH = CN()
    cfg.TRAINER.MLPHEGRAPH.FM_NAMES          = ["biomedclip", "plip", "conch"]
    cfg.TRAINER.MLPHEGRAPH.COMMON_DIM        = 512
    cfg.TRAINER.MLPHEGRAPH.HIDDEN_DIM        = 512
    cfg.TRAINER.MLPHEGRAPH.DROPOUT           = 0.1
    cfg.TRAINER.MLPHEGRAPH.BETA_P2N          = 0.5
    cfg.TRAINER.MLPHEGRAPH.BETA_V2N          = 0.5
    cfg.TRAINER.MLPHEGRAPH.ALPHA_P2P         = 0.06
    cfg.TRAINER.MLPHEGRAPH.ALPHA_V2P         = 0.24
    cfg.TRAINER.MLPHEGRAPH.ALPHA_N2P_TRAIN   = 0.2
    cfg.TRAINER.MLPHEGRAPH.ALPHA_N2P_TEST    = 0.1
    cfg.TRAINER.MLPHEGRAPH.GAMMA             = 0.5
    cfg.TRAINER.MLPHEGRAPH.CACHE_RATIO       = 0.1
    cfg.TRAINER.MLPHEGRAPH.CACHE_OUTPUT      = 1.0
    cfg.TRAINER.MLPHEGRAPH.LOGIT_SCALE_INIT  = 2.3
    cfg.TRAINER.MLPHEGRAPH.PREC              = "fp32"

    cfg.MODEL.NUM_SAMPLE_EPOCHS = 10  # epochs for prototype extraction

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    """Setup config from files and CLI arguments."""
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments (key-value pairs)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


# =============================================================================
# Main
# =============================================================================

def main(args):
    cfg = setup_cfg(args)
    
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print(f"** System info **\n{collect_env_info()}\n")

    # Build trainer
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Foundation Model Fusion Training"
    )
    
    # Data
    parser.add_argument(
        "--root", 
        type=str, 
        default="", 
        help="Path to dataset root"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="", 
        help="Output directory for checkpoints and logs"
    )
    
    # Training control
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint directory to resume from",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=-1, 
        help="Random seed (positive value enables fixed seed)"
    )
    parser.add_argument(
        "--no-train", 
        action="store_true", 
        help="Skip training, only run evaluation"
    )
    
    # Domain adaptation (optional)
    parser.add_argument(
        "--source-domains", 
        type=str, 
        nargs="+", 
        help="Source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", 
        type=str, 
        nargs="+", 
        help="Target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", 
        type=str, 
        nargs="+", 
        help="Data augmentation methods"
    )
    
    # Config files
    parser.add_argument(
        "--config-file", 
        type=str, 
        default="", 
        help="Path to trainer config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="Path to dataset config file",
    )
    
    # Model
    parser.add_argument(
        "--trainer", 
        type=str, 
        default="MultiFMFusion", 
        help="Trainer name"
    )
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="", 
        help="Backbone name (for single-FM baselines)"
    )
    parser.add_argument(
        "--head", 
        type=str, 
        default="", 
        help="Head name"
    )
    parser.add_argument(
        "--init_weights", 
        type=str, 
        default="", 
        help="Path to pretrained weights"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval-only", 
        action="store_true", 
        help="Run evaluation only"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Directory to load model for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", 
        type=int, 
        help="Load model from specific epoch"
    )
    
    # Additional config overrides
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options via command-line (KEY VALUE pairs)",
    )
    
    args = parser.parse_args()
    main(args)
