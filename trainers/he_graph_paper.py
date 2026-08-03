"""
HeGraphAdapter — paper-faithful implementation.
Changes vs. the original he_graph.py (see discrepancy report):

1. compute_adj(negative=True) now only negates diagonal entries;
   off-diagonal entries stay as positive cosine similarity (Sec 3.1.1).
2. GraphModel uses a single W per layer instead of two separate matrices
   (Eq. 1, 3, 5).
3. No learnable bias in the GCN (not in paper equations).
4. Visual node aggregation uses self_add=False (Eq. 5 has no self-term).
5. Meta-path weights (α, β, γ) are fixed hyperparameters, not learnable
   nn.Parameters (Sec 4.1 / Table 3 shows learnable is worse).
6. α_{n→p} uses a different (smaller) value at test time
   ("we set the fusion weight of meta-path n→p to 0.1 during testing").
7. Visual-based classifier follows Tip-Adapter formulation (Eq. 8):
   A = exp(sim(z, X_cache) − 1) · L  using all CK per-sample features
   and one-hot labels, instead of class-mean dot-product.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.metrics import compute_accuracy
from torch.cuda.amp import GradScaler, autocast
from transformers import CLIPModel

from clip import clip
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from trainers.baseclip_graph_v1 import CUSTOM_TEMPLATES
from trainers.model_registry import load_clip_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_negative_prompt(text):
    """Paper: 'A photo of no {class}'."""
    if " of a " in text:
        output_text = text.replace(" of a ", " of no ")
    elif " of " in text:
        output_text = text.replace(" of ", " of no ")
    else:
        output_text = "a photo of no " + text
    print("negative text:", output_text)
    return output_text


def compute_adj(x, y, p=2, dim=2, negative=False):
    """Compute cosine-similarity adjacency matrix.

    When *negative=True* (for p→n and v→n edges):
      - diagonal entries are **negated** (opposite information, Sec 3.1.1)
      - off-diagonal entries remain raw cosine similarity

    No clamping: the paper (Sec 3.1.1) uses the cosine similarity directly as the
    edge weight, so negative (anti-correlated) edges are kept. The GCN degree term
    uses abs(adj), so signed edges are handled correctly.
    """
    x = F.normalize(x, p=p, dim=dim)          # [1, N, D]
    y = F.normalize(y, p=p, dim=dim)           # [1, N, D]
    A = torch.bmm(x, y.transpose(1, 2))       # [1, N, N]

    if negative:
        # Negate only the diagonal (same-class) entries  (Fix #1)
        mask = torch.eye(A.shape[1], A.shape[2], device=A.device).unsqueeze(0)  # [1,N,N]
        A = (1.0 - mask) * A + mask * (-A)
    return A


# ---------------------------------------------------------------------------
# Paper-faithful GCN layer  (Eq. 1 / 3 / 5)
# ---------------------------------------------------------------------------

class GraphModel(nn.Module):
    """Single-weight GCN layer without bias (matches Eq. 1, 3, 5).

    m_i = tanh( 1/(d_i+1) · W·x_i  +  Σ_j  r_ij / √((d_i+1)(d_j+1)) · W·x_j )

    *self_add*  controls whether the first (self-loop) term is included.
    Eq. 5 (visual aggregation) has no self-loop → pass self_add=False.

    Activation: tanh (not ReLU). HeGraphAdapter extends GraphAdapter, whose GCN
    uses tanh, and the HeGraphAdapter paper does not specify the activation; tanh
    is also sign-preserving, keeping the dissimilarity signal on the sign-flipped
    edges into the negative-text nodes that ReLU would clip.
    """

    def __init__(self, hidden_dim=1024, class_num=None):
        super().__init__()
        self.act = nn.Tanh()
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, x, y, adj, self_add=True):
        """
        Args:
            x: [1, N, D]  — the node being aggregated INTO (self-feature)
            y: [1, N, D]  — the neighbour features (may equal x)
            adj: [1, N, N] — edge-weight matrix r_{ij}
            self_add: whether to include the 1/(d_i+1) · W·x_i self-loop term
        """
        # degree = Σ_j |r_ij|   (paper: d_i = Σ_{j∈N_i} |r_ij|)
        degree = torch.abs(adj).sum(-1, keepdim=True) + 1       # [1,N,1]
        sqrt_deg = torch.sqrt(degree)                           # [1,N,1]
        d_denom = torch.bmm(sqrt_deg, sqrt_deg.transpose(1, 2))  # [1,N,N]

        Wy = torch.matmul(y, self.W)                            # [1,N,D]
        neighbour_agg = torch.bmm(adj / d_denom, Wy)            # [1,N,D]

        if self_add:
            Wx = torch.matmul(x, self.W)                        # [1,N,D]
            output = (1.0 / degree) * Wx + neighbour_agg
        else:
            output = neighbour_agg

        return self.act(output)


# ---------------------------------------------------------------------------
# HeGraph learner  (Sec 3.1.2)
# ---------------------------------------------------------------------------

class GraphLearner(nn.Module):
    """Heterogeneous graph adapter with three aggregation steps.

    Meta-path weights are **fixed** hyperparameters (Table 3 / Sec 4.1).
    Default values correspond to the optimal ones reported for ImageNet:
        β_{p→n} = 0.5,  β_{v→n} = 0.5       (Eq. 2)
        α_{p→p} = 0.06, α_{v→p} = 0.24      (Eq. 4)
        α_{n→p} = 0.2 (train) / 0.1 (test)   (Eq. 4  + Sec 4.1)
        γ = 0.5                                (Eq. 6)
    """

    def __init__(self, cfg, classnames, clip_model,
                 base_text_features, base_negative_features,
                 base_img_features, base_img_features_all, base_labels_onehot):
        super().__init__()
        self.device = clip_model.device

        # Frozen base features (registered as buffers — not trainable)
        self.register_buffer("base_text_features", base_text_features)
        self.register_buffer("base_negative_features", base_negative_features)
        self.register_buffer("base_img_features", base_img_features)            # [C, D]  class-means for graph
        self.register_buffer("base_img_features_all", base_img_features_all)    # [CK, D] all per-sample features
        self.register_buffer("base_labels_onehot", base_labels_onehot)          # [CK, C] one-hot labels

        print(">> BASE features: ", base_text_features.shape,
              base_negative_features.shape, base_img_features.shape,
              "| cache:", base_img_features_all.shape, base_labels_onehot.shape)

        hidden_dim = base_text_features.shape[-1]
        num_classes = base_text_features.shape[0]

        # Three GCN layers — one W each (Fix #2, #3)
        self.gcn_neg_agg = GraphModel(hidden_dim, class_num=num_classes)
        self.gcn_pos_agg = GraphModel(hidden_dim, class_num=num_classes)
        self.gcn_vis_agg = GraphModel(hidden_dim, class_num=num_classes)

        # ----- Fixed meta-path weights (Fix #5) -----
        # Read from config or fall back to paper defaults
        hegraph_cfg = getattr(cfg.TRAINER, 'HEGRAPH', None)

        self.beta_p2n  = getattr(hegraph_cfg, 'BETA_P2N', 0.5)   if hegraph_cfg else 0.5
        self.beta_v2n  = getattr(hegraph_cfg, 'BETA_V2N', 0.5)   if hegraph_cfg else 0.5
        self.alpha_p2p = getattr(hegraph_cfg, 'ALPHA_P2P', 0.06) if hegraph_cfg else 0.06
        self.alpha_v2p = getattr(hegraph_cfg, 'ALPHA_V2P', 0.24) if hegraph_cfg else 0.24
        self.alpha_n2p_train = getattr(hegraph_cfg, 'ALPHA_N2P_TRAIN', 0.2) if hegraph_cfg else 0.2
        self.alpha_n2p_test  = getattr(hegraph_cfg, 'ALPHA_N2P_TEST',  0.1) if hegraph_cfg else 0.1   # Fix #7
        self.gamma     = getattr(hegraph_cfg, 'GAMMA', 0.5)      if hegraph_cfg else 0.5

    def forward(self):
        num_classes = self.base_text_features.shape[0]
        text_features = self.base_text_features.unsqueeze(0)           # [1,C,D]
        negative_features = self.base_negative_features.unsqueeze(0)   # [1,C,D]
        vis_features = self.base_img_features.unsqueeze(0)             # [1,C,D]

        # -------- Compute ALL adjacency matrices FIRST (from frozen base features) --------
        # This ensures the graph structure is static and doesn't change during forward pass
        pos_neg_adj = compute_adj(negative_features, text_features, negative=True)   # for Eq. 1
        vis_neg_adj = compute_adj(negative_features, vis_features, negative=True)    # for Eq. 1
        pos_pos_adj = compute_adj(text_features, text_features)                      # for Eq. 3
        pos_pos_adj[0, range(num_classes), range(num_classes)] = 0.0                 # remove self-loops
        vis_pos_adj = compute_adj(text_features, vis_features)                       # for Eq. 3
        neg_pos_adj = compute_adj(text_features, negative_features, negative=True)   # for Eq. 3 (from BASE neg features!)
        vis_vis_adj = compute_adj(vis_features, vis_features)                        # for Eq. 5
        vis_vis_adj[0, range(num_classes), range(num_classes)] = 0.0                 # remove self-loops

        # -------- 1) Negative text node aggregation (Eq. 1-2) --------
        pos_neg_output = self.gcn_neg_agg(negative_features, text_features, pos_neg_adj, self_add=True)
        vis_neg_output = self.gcn_neg_agg(negative_features, vis_features, vis_neg_adj, self_add=True)
        negative_features = negative_features + self.beta_p2n * pos_neg_output + self.beta_v2n * vis_neg_output

        # -------- 2) Positive text node aggregation (Eq. 3-4) --------

        pos_pos_output = self.gcn_pos_agg(text_features, text_features, pos_pos_adj, self_add=True)
        vis_pos_output = self.gcn_pos_agg(text_features, vis_features, vis_pos_adj, self_add=True)
        neg_pos_output = self.gcn_pos_agg(text_features, negative_features, neg_pos_adj, self_add=True)

        # Use different α_{n→p} at train vs test time (Fix #7)
        alpha_n2p = self.alpha_n2p_train if self.training else self.alpha_n2p_test

        text_features = (text_features
                         + self.alpha_p2p * pos_pos_output
                         + self.alpha_v2p * vis_pos_output
                         + alpha_n2p * neg_pos_output)

        # -------- 3) Visual node aggregation (Eq. 5-6) --------
        # Eq. 5: NO self-connection term → self_add=False  (Fix #4)
        vis_vis_output = self.gcn_vis_agg(vis_features, vis_features, vis_vis_adj, self_add=False)

        # Eq. 6: x_cache_i ← x_cache_i + γ · m_bias_i
        # Apply the learned visual bias to ALL per-sample cache features
        # vis_vis_output is [1,C,D] — one bias vector per class
        vis_bias = vis_vis_output.squeeze(0)  # [C, D]

        return text_features.squeeze(0), vis_bias


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _get_base_text_features(cfg, classnames, clip_model, model_wrapper):
    device = next(model_wrapper.parameters()).device
    if clip_model.dtype == torch.float16:
        model_wrapper = model_wrapper.cuda()
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        negative_embeddings = []
        for text in classnames:
            texts = [template.format(text) for template in TEMPLATES]
            tokens = model_wrapper.tokenize(texts).to(device)
            text_embedding = model_wrapper.encode_text(tokens)
            text_embeddings.append(text_embedding)

            neg_texts = [get_negative_prompt(template.format(text)) for template in TEMPLATES]
            neg_tokens = model_wrapper.tokenize(neg_texts).to(device)
            text_embedding = model_wrapper.encode_text(neg_tokens)
            negative_embeddings.append(text_embedding)

    text_embeddings = torch.stack(text_embeddings).mean(1)
    negative_embeddings = torch.stack(negative_embeddings).mean(1)
    model_wrapper = model_wrapper.to(device)
    return text_embeddings.to(device), negative_embeddings.to(device)


def _get_base_image_features(cfg, classnames, clip_model, model_wrapper, train_loader_x):
    """Return *both* class-mean features (for the graph) AND all per-sample
    features + one-hot labels (for the Tip-Adapter cache, Eq. 8).
    
    Uses a custom DataLoader with drop_last=False to ensure ALL samples are
    captured for the cache, regardless of batch size settings.
    """
    from torch.utils.data import DataLoader
    
    device = next(model_wrapper.parameters()).device
    if clip_model.dtype == torch.float16:
        model_wrapper = model_wrapper.cuda()
    num_sample_epochs = cfg.MODEL.NUM_SAMPLE_EPOCHS
    num_classes = len(classnames)
    
    # Create a new loader with drop_last=False to capture ALL samples
    # This fixes the issue where Dassl's DataManager drops the last incomplete batch
    dataset = train_loader_x.dataset
    feature_loader = DataLoader(
        dataset,
        batch_size=train_loader_x.batch_size,
        shuffle=False,  # No shuffle needed for feature extraction
        num_workers=train_loader_x.num_workers,
        drop_last=False,  # CRITICAL: capture all samples
        pin_memory=True,
    )
    print(f"Feature extraction: created loader with drop_last=False "
          f"(dataset size: {len(dataset)}, batch_size: {feature_loader.batch_size})")

    with torch.no_grad():
        img_feature = []
        labels = []
        for epoch in range(num_sample_epochs):
            for batch_idx, batch in enumerate(feature_loader):
                image = batch["img"].cuda()
                label = batch["label"].cuda()
                image_features = model_wrapper.encode_image(image.type(clip_model.dtype)).detach()
                img_feature.append(image_features)
                labels.append(label)

        img_feature_list = torch.cat(img_feature, dim=0)   # [CK, D]
        label_list = torch.cat(labels, dim=0)               # [CK]

        # Sort by label for consistent ordering
        sorted_labels, indices = torch.sort(label_list)
        img_feature_sorted = torch.index_select(img_feature_list, 0, indices)

        # --- All per-sample features for cache (Eq. 8) ---
        cache_features_all = img_feature_sorted                                # [CK, D]
        labels_onehot = F.one_hot(sorted_labels, num_classes=num_classes).float()  # [CK, C]

        # --- Class-mean features for graph nodes ---
        # Handle uneven class distributions by computing mean per class
        b, d = img_feature_sorted.size()
        class_mean_features = torch.zeros(num_classes, d, device=img_feature_sorted.device, dtype=img_feature_sorted.dtype)
        
        for cls_idx in range(num_classes):
            mask = (sorted_labels == cls_idx)
            if mask.sum() > 0:
                class_mean_features[cls_idx] = img_feature_sorted[mask].mean(dim=0)

        model_wrapper = model_wrapper.to(device)

    return (class_mean_features.to(device),
            cache_features_all.to(device),
            labels_onehot.to(device))


# ---------------------------------------------------------------------------
# CustomCLIP  (Sec 3.2)
# ---------------------------------------------------------------------------

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, train_loader_x, cache_output=1.0):
        super().__init__()
        self.clip_model = clip_model

        base_text_features, base_negative_features = _get_base_text_features(
            cfg, classnames, clip_model, clip_model)
        class_mean_features, cache_features_all, labels_onehot = _get_base_image_features(
            cfg, classnames, clip_model, clip_model, train_loader_x)

        print(f"Num classes: {len(classnames)}, "
              f"Class-mean features: {class_mean_features.shape}, "
              f"Cache features: {cache_features_all.shape}")

        self.hegraph_learner = GraphLearner(
            cfg, classnames, clip_model,
            base_text_features, base_negative_features,
            class_mean_features, cache_features_all, labels_onehot,
        )

        self.cache_output = cache_output
        # Store original cache features to apply visual bias (Eq. 6)
        self.register_buffer("cache_features_all", cache_features_all)   # [CK, D]
        self.register_buffer("labels_onehot", labels_onehot)             # [CK, C]

        # Read gamma from the learner so we can apply bias in forward
        self.gamma = self.hegraph_learner.gamma

    def forward(self, image, batch=None, is_training=False):
        try:
            image_features = self.clip_model.encode_image(image.type(self.clip_model.dtype)).detach()
        except Exception:
            image_features = self.clip_model.encode_image(image.float()).detach()

        # Graph adapter produces updated text features and per-class visual bias
        text_features, vis_bias = self.hegraph_learner()  # [C,D], [C,D]

        # --- Apply visual bias to per-sample cache features (Eq. 6) ---
        # vis_bias is [C, D]; map each sample to its class bias via labels_onehot [CK, C]
        per_sample_bias = self.labels_onehot @ vis_bias           # [CK, D]
        cache_features = self.cache_features_all + self.gamma * per_sample_bias  # [CK, D]

        # Normalise
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)
        cache_features = F.normalize(cache_features, dim=-1)

        logit_scale = self.clip_model.logit_scale.exp()

        # --- Text-based classifier (Eq. 7) ---
        logits_text = logit_scale * image_features @ text_features.t()      # [B, C]

        # --- Visual-based classifier / Tip-Adapter (Eq. 8) ---
        # A = exp(sim(z, X_cache) − 1) · L,  then scale by τ for Lvisual
        sim_cache = image_features @ cache_features.t()                     # [B, CK]
        A = torch.exp(sim_cache - 1.0) @ self.labels_onehot                 # [B, C]
        logits_cache = logit_scale * A                                      # [B, C] — apply τ (Eq. 8)

        if not is_training:
            output = (F.softmax(logits_text, dim=-1)
                      + self.cache_output * F.softmax(logits_cache, dim=-1)) / (1.0 + self.cache_output)
            return output

        return logits_text, logits_cache


# ---------------------------------------------------------------------------
# Trainer (Dassl)
# ---------------------------------------------------------------------------

@TRAINER_REGISTRY.register()
class HeGraphCLIPPaper(TrainerX):
    """Trainer registered under a separate name so it can coexist with the
    original HeGraphCLIP without conflicts."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Paper: λ = 0.1
        self.cache_ratio  = getattr(cfg.TRAINER.HEGRAPH, 'CACHE_RATIO',  0.1)
        self.cache_output = getattr(cfg.TRAINER.HEGRAPH, 'CACHE_OUTPUT', 1.0)

        print(f"Loading backbone: {cfg.MODEL.BACKBONE.NAME}")
        clip_model = load_clip_model(cfg, device="cpu")
        clip_model.to(self.device)

        if cfg.TRAINER.COOP.PREC in ("fp32", "amp"):
            clip_model.float()

        print("Building custom CLIP (paper-faithful)")
        self.model = CustomCLIP(
            cfg, classnames, clip_model, self.train_loader_x,
            cache_output=self.cache_output,
        ).cuda()

        # Freeze everything except the three GCN weight matrices
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "hegraph_learner" not in name:
                param.requires_grad_(False)

        # Only W^n, W^p, W^v inside the GCN layers are trainable
        for param in self.model.hegraph_learner.parameters():
            param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.hegraph_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.float()

        self.optim = build_optimizer(model=self.model.hegraph_learner, optim_cfg=cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("hegraph_learner", self.model.hegraph_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, batch = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                output_text, output_cache = self.model(image, batch, is_training=True)
                loss = (F.cross_entropy(output_text, label)
                        + self.cache_ratio * F.cross_entropy(output_cache, label))
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_text, output_cache = self.model(image, batch, is_training=True)
            loss = (F.cross_entropy(output_text, label)
                    + self.cache_ratio * F.cross_entropy(output_cache, label))
            self.model_backward_and_update(loss)

        output = (F.softmax(output_text, dim=-1)
                  + self.cache_output * F.softmax(output_cache, dim=-1)) / (1.0 + self.cache_output)
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def test(self, split=None):
        """Override test() to also report Cohen's Kappa and Macro AUC."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)
                all_logits.append(output.cpu())
                all_labels.append(label.cpu())

        results = self.evaluator.evaluate()

        y_true = np.concatenate([l.numpy() for l in all_labels])
        logits = torch.cat(all_logits)
        y_pred = logits.argmax(dim=1).numpy()
        y_prob = torch.softmax(logits, dim=1).numpy()
        num_classes = y_prob.shape[1]

        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            multi_class = "ovr" if num_classes > 2 else "raise"
            auc = roc_auc_score(y_true, y_prob, multi_class=multi_class, average="macro")
        except Exception:
            auc = float("nan")

        print(f"* kappa: {kappa:.4f}")
        print(f"* macro_auc: {auc:.4f}")

        results["kappa"] = kappa
        results["macro_auc"] = auc

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = os.path.join(directory, name, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        """Save checkpoint every 10 epochs."""
        # Call parent's after_epoch if it exists
        super().after_epoch()
        
        # Save every 10 epochs (and always save final epoch)
        if (self.epoch + 1) % 10 == 0 or (self.epoch + 1) == self.max_epoch:
            self.save_model(self.epoch, self.output_dir)
