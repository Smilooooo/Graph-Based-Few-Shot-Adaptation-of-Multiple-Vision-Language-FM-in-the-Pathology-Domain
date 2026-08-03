"""
MLPHeGraphAdapter Trainer – Early MLP Fusion + HeGraph Heterogeneous GCN.

Extends HeGraphAdapter (He et al., 2024) to use all three pathology VLMs
(BiomedCLIP, PLIP, CONCH) simultaneously via early MLP fusion, following the
same multi-backbone strategy as MLPGraphAdapter.

Pipeline
────────
1. Extract features from all 3 frozen FM encoders.
2. JointAdapter MLP: concat(3×512=1536) → 512-dim shared space.
3. HeGraph 3-step heterogeneous GCN refines positive text prototypes and
   produces a per-class visual bias for the Tip-Adapter cache.
4. Dual classification: text logits + visual cache logits (Eq. 7-8).
5. Combined softmax prediction at test time.

Usage
─────
python train.py \\
    --trainer MLPHeGraphAdapter \\
    --config-file configs/trainers/mlphegraph.yaml \\
    --dataset-config-file configs/datasets/lunghist_descriptive.yaml \\
    --root /path/to/datasets \\
    --output-dir output/mlphegraph/lunghist_4shot_seed1 \\
    DATASET.NUM_SHOTS 4 SEED 1
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import cohen_kappa_score, roc_auc_score

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint

from trainers.model_registry import MODEL_CONFIGS, load_clip_model
from trainers.mlphegraph_model import MLPHeGraphAdapterModel


# ── Per-FM input resolutions ───────────────────────────────────────────────────
FM_INPUT_SIZES: dict = {
    "biomedclip": 224,
    "plip":       224,
    "conch":      448,
}

# ── Dataset-specific text-prompt templates ─────────────────────────────────────
CUSTOM_TEMPLATES: dict = {
    "LungHistDescriptive":  "a histological image of a {} cell in pulmonary pathology.",
    "BRACSDescriptive":     "a histological image of a {} cell in breast carcinoma subtyping.",
    "ICIAR2018Descriptive": "a histopathological image of {}.",
}
DEFAULT_TEMPLATE = "a histological image of a {} cell."


def _get_negative_prompt(text: str) -> str:
    """Convert a positive prompt to a negative one (paper: 'a photo of no {class}')."""
    if " of a " in text:
        return text.replace(" of a ", " of no ")
    if " of " in text:
        return text.replace(" of ", " of no ")
    return "a photo of no " + text


# ── Minimal cfg shim for load_clip_model ──────────────────────────────────────

class _MockCfg:
    class _BB:
        def __init__(self, name): self.NAME = name
    class _M:
        def __init__(self, name): self.BACKBONE = _MockCfg._BB(name)
    def __init__(self, fm_name): self.MODEL = _MockCfg._M(fm_name)


# =============================================================================
# Trainer
# =============================================================================

@TRAINER_REGISTRY.register()
class MLPHeGraphAdapter(TrainerX):
    """
    MLPHeGraphAdapter Trainer.

    Combines:
    • JointAdapter MLP (early fusion of 3-FM concatenated features, 1536→512)
    • HeGraph heterogeneous GCN (3 node types: positive text, negative text, visual;
      3-step sequential message passing with fixed meta-path weights)
    • Dual classification: text cosine logits + Tip-Adapter visual cache logits
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MLPHEGRAPH.PREC in ("fp32", "amp"), \
            "TRAINER.MLPHEGRAPH.PREC must be 'fp32' or 'amp'"

    # ── Model construction ─────────────────────────────────────────────────────

    def build_model(self):
        cfg    = self.cfg
        hg     = cfg.TRAINER.MLPHEGRAPH
        device = self.device

        self.fm_names:    List[str] = list(hg.FM_NAMES)
        self.num_fms:     int       = len(self.fm_names)
        self.num_classes: int       = self.dm.num_classes
        self.class_names: List[str] = self.dm.dataset.classnames
        self.prec:        str       = hg.PREC
        self.cache_ratio: float     = hg.CACHE_RATIO

        print("=" * 62)
        print("Building MLPHeGraphAdapter (MLP fusion + HeGraph GCN)")
        print(f"  FMs           : {self.fm_names}")
        print(f"  Common dim    : {hg.COMMON_DIM}")
        print(f"  Hidden dim    : {hg.HIDDEN_DIM}")
        print(f"  beta_p2n/v2n : {hg.BETA_P2N}/{hg.BETA_V2N}  (fixed)")
        print(f"  alpha_p2p/v2p: {hg.ALPHA_P2P}/{hg.ALPHA_V2P}  (fixed)")
        print(f"  alpha_n2p train/test: {hg.ALPHA_N2P_TRAIN}/{hg.ALPHA_N2P_TEST}  (fixed)")
        print(f"  gamma / cache_ratio / cache_output: "
              f"{hg.GAMMA} / {hg.CACHE_RATIO} / {hg.CACHE_OUTPUT}")
        print(f"  Precision     : {self.prec}")
        print(f"  Num classes   : {self.num_classes}")
        print("=" * 62)

        # ── Load frozen FM encoders ──────────────────────────────────────────
        self.encoders:  List = []
        self.feat_dims: List[int] = []
        for fm in self.fm_names:
            print(f"  Loading {fm} ...")
            enc = load_clip_model(_MockCfg(fm), device="cpu")
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            self.encoders.append(enc)
            self.feat_dims.append(MODEL_CONFIGS[fm]["feature_dim"])

        feat_dim: int = self.feat_dims[0]  # all 512 in this project

        # ── Extract all features ─────────────────────────────────────────────
        print("\nExtracting positive and negative text features ...")
        raw_text_cat, raw_neg_cat = self._extract_multi_fm_text_features()
        # raw_text_cat : [C, 1536], raw_neg_cat : [C, 1536]

        print("Extracting visual class-mean and per-sample cache features ...")
        raw_vis_cat, raw_cache_cat, labels_onehot = \
            self._extract_multi_fm_image_features()
        # raw_vis_cat   : [C,  1536]
        # raw_cache_cat : [CK, 1536]   CK = num_sample_epochs × shots × num_classes
        # labels_onehot : [CK, C]

        print(f"  Cache size: {raw_cache_cat.shape[0]} samples x "
              f"{raw_cache_cat.shape[1]} features")

        # ── Build model ──────────────────────────────────────────────────────
        self.model = MLPHeGraphAdapterModel(
            num_fms          = self.num_fms,
            feat_dim         = feat_dim,
            hidden_dim       = hg.HIDDEN_DIM,
            common_dim       = hg.COMMON_DIM,
            num_classes      = self.num_classes,
            beta_p2n         = hg.BETA_P2N,
            beta_v2n         = hg.BETA_V2N,
            alpha_p2p        = hg.ALPHA_P2P,
            alpha_v2p        = hg.ALPHA_V2P,
            alpha_n2p_train  = hg.ALPHA_N2P_TRAIN,
            alpha_n2p_test   = hg.ALPHA_N2P_TEST,
            gamma            = hg.GAMMA,
            cache_output     = hg.CACHE_OUTPUT,
            dropout          = hg.DROPOUT,
            logit_scale_init = hg.LOGIT_SCALE_INIT,
        ).to(device)

        # ── Register raw feature tensors as frozen buffers on the model ──────
        self.model.register_buffer("raw_text_cat",  raw_text_cat.float().to(device))
        self.model.register_buffer("raw_neg_cat",   raw_neg_cat.float().to(device))
        self.model.register_buffer("raw_vis_cat",   raw_vis_cat.float().to(device))
        self.model.register_buffer("raw_cache_cat", raw_cache_cat.float().to(device))
        self.model.register_buffer("labels_onehot", labels_onehot.float().to(device))

        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        print(f"\nTrainable parameters: {trainable:,}")

        # ── Optimizer + scheduler ────────────────────────────────────────────
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("mlphegraph_model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if self.prec == "amp" else None

    # ── Feature extraction helpers ─────────────────────────────────────────────

    def _img_for_fm(self, imgs: torch.Tensor, fm_name: str) -> torch.Tensor:
        """Resize image batch to the resolution expected by the given FM."""
        target = FM_INPUT_SIZES[fm_name]
        if imgs.shape[-1] != target:
            imgs = TF.resize(imgs, [target, target], antialias=True)
        return imgs

    @torch.no_grad()
    def _extract_features(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract L2-normalised image features from all frozen FM encoders.

        imgs : [B, 3, H, W]
        out  : list[num_fms] of [B, feat_dim]
        """
        feats = []
        for enc, fm in zip(self.encoders, self.fm_names):
            x = self._img_for_fm(imgs, fm)
            try:
                f = enc.encode_image(x.type(enc.dtype))
            except Exception:
                f = enc.encode_image(x.float())
            feats.append(F.normalize(f.float(), dim=-1))
        return feats

    @torch.no_grad()
    def _extract_multi_fm_text_features(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode positive and negative text prompts with each FM, then concatenate.

        Returns
        ───────
        raw_text_cat : [C, num_fms × feat_dim]  positive text, multi-FM concatenated
        raw_neg_cat  : [C, num_fms × feat_dim]  negative text, multi-FM concatenated
        """
        dataset_name = self.cfg.DATASET.NAME
        template     = CUSTOM_TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)

        pos_per_fm: List[torch.Tensor] = []  # each [C, feat_dim]
        neg_per_fm: List[torch.Tensor] = []

        for enc, fm in zip(self.encoders, self.fm_names):
            pos_prompts = [template.format(cn) for cn in self.class_names]
            neg_prompts = [_get_negative_prompt(p) for p in pos_prompts]

            def _encode(prompts):
                if hasattr(enc, "tokenize"):
                    tokens = enc.tokenize(prompts).to(self.device)
                else:
                    from clip import clip as openai_clip
                    tokens = openai_clip.tokenize(prompts).to(self.device)
                feats = enc.encode_text(tokens)
                return F.normalize(feats.float(), dim=-1)

            pos_per_fm.append(_encode(pos_prompts).cpu())
            neg_per_fm.append(_encode(neg_prompts).cpu())

        raw_text_cat = torch.cat(pos_per_fm, dim=-1)  # [C, 1536]
        raw_neg_cat  = torch.cat(neg_per_fm, dim=-1)  # [C, 1536]
        return raw_text_cat, raw_neg_cat

    @torch.no_grad()
    def _extract_multi_fm_image_features(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract visual features from all 3 FMs across NUM_SAMPLE_EPOCHS augmented
        passes over the support set, then concatenate across FMs.

        Uses drop_last=False to ensure ALL support samples enter the cache,
        avoiding the class imbalance that drop_last=True can introduce.

        Returns
        ───────
        raw_vis_cat   : [C,  1536]  per-class mean of concatenated features
        raw_cache_cat : [CK, 1536]  all per-sample concatenated features
        labels_onehot : [CK, C]     one-hot labels for each cached sample
        """
        num_sample_epochs = self.cfg.MODEL.NUM_SAMPLE_EPOCHS

        feature_loader = DataLoader(
            self.train_loader_x.dataset,
            batch_size  = self.train_loader_x.batch_size,
            shuffle     = False,
            num_workers = 0,
            drop_last   = False,
            pin_memory  = True,
        )
        print(f"  Cache extraction: {num_sample_epochs} epochs x "
              f"{len(feature_loader.dataset)} samples (drop_last=False)")

        all_feats:  List[torch.Tensor] = []  # [B, 1536] per step
        all_labels: List[torch.Tensor] = []  # [B] per step

        for _ in range(num_sample_epochs):
            for batch in feature_loader:
                imgs   = batch["img"].to(self.device)
                labels = batch["label"]

                per_fm = self._extract_features(imgs)          # list of [B, feat_dim]
                cat_f  = torch.cat(per_fm, dim=-1).cpu()       # [B, 1536]
                all_feats.append(cat_f)
                all_labels.append(labels)

        all_feats_t  = torch.cat(all_feats,  dim=0)   # [CK, 1536]
        all_labels_t = torch.cat(all_labels, dim=0)   # [CK]

        # Sort by label for consistent ordering
        sorted_labels, idx = torch.sort(all_labels_t)
        all_feats_sorted   = all_feats_t[idx]          # [CK, 1536]

        # One-hot labels
        labels_onehot = F.one_hot(
            sorted_labels, num_classes=self.num_classes
        ).float()                                       # [CK, C]

        # Per-class mean for the graph visual nodes
        raw_vis_cat = torch.zeros(
            self.num_classes, all_feats_sorted.shape[1],
            dtype=all_feats_sorted.dtype,
        )
        for c in range(self.num_classes):
            mask = (sorted_labels == c)
            if mask.sum() > 0:
                raw_vis_cat[c] = all_feats_sorted[mask].mean(dim=0)

        return raw_vis_cat, all_feats_sorted, labels_onehot

    # ── Training step ──────────────────────────────────────────────────────────

    def forward_backward(self, batch):
        imgs, labels = self.parse_batch_train(batch)

        # Extract per-FM features and concatenate → [B, 1536]
        query_feats = self._extract_features(imgs)             # list of [B, 512]
        query_cat   = torch.cat(query_feats, dim=-1)           # [B, 1536]

        if self.prec == "amp":
            with autocast():
                logits_text, logits_cache = self.model(query_cat, is_training=True)
                loss = (F.cross_entropy(logits_text, labels)
                        + self.cache_ratio * F.cross_entropy(logits_cache, labels))
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits_text, logits_cache = self.model(query_cat, is_training=True)
            loss = (F.cross_entropy(logits_text, labels)
                    + self.cache_ratio * F.cross_entropy(logits_cache, labels))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # Accuracy on combined softmax for logging
        hg = self.cfg.TRAINER.MLPHEGRAPH
        combined = (F.softmax(logits_text, dim=-1)
                    + hg.CACHE_OUTPUT * F.softmax(logits_cache, dim=-1)) / (1.0 + hg.CACHE_OUTPUT)
        return {
            "loss": loss.item(),
            "acc":  compute_accuracy(combined, labels)[0].item(),
        }

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def model_inference(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Run inference, returning combined softmax probabilities [B, C].
        """
        query_feats = self._extract_features(imgs)
        query_cat   = torch.cat(query_feats, dim=-1)
        return self.model(query_cat, is_training=False)

    # ── Evaluation (extended with kappa + macro-AUC) ───────────────────────────

    def test(self, split=None):
        """Evaluate and additionally report Cohen's Kappa and macro AUC."""
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

        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)
                all_logits.append(output.cpu())
                all_labels.append(label.cpu())

        results  = self.evaluator.evaluate()
        y_true   = np.concatenate([l.numpy() for l in all_labels])
        logits   = torch.cat(all_logits)
        y_pred   = logits.argmax(dim=1).numpy()
        y_prob   = torch.softmax(logits, dim=1).numpy()
        n_cls    = y_prob.shape[1]

        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            multi_class = "ovr" if n_cls > 2 else "raise"
            auc = roc_auc_score(y_true, y_prob, multi_class=multi_class,
                                average="macro")
        except Exception:
            auc = float("nan")

        print(f"* kappa: {kappa:.4f}")
        print(f"* macro_auc: {auc:.4f}")

        results["kappa"]     = kappa
        results["macro_auc"] = auc

        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)

        return list(results.values())[0]

    def after_epoch(self):
        """Evaluate on the test set and checkpoint every 10th epoch.

        Mirrors the MLPGA / MLP-Fusion trainers so every method logs test-set
        metrics every 10 epochs (in-training eval rather than reloading
        checkpoints afterwards, which is too slow).  The intermediate blocks are
        parsed into results.jsonl by the multi-seed runner.  The final epoch is
        evaluated by ``after_train()``, so it is skipped below.  A checkpoint is
        also saved every 10 epochs so a specific epoch can be reloaded later.
        """
        epoch = self.epoch + 1
        last_epoch = epoch == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_freq = epoch % 10 == 0

        if do_test and meet_freq and not last_epoch:
            print(f"\n=> Intermediate test evaluation at epoch {epoch}")
            self.test(split="test")
            self.set_model_mode("train")

        if meet_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def load_model(self, directory, epoch=None):
        if not directory:
            print("load_model() skipped - no directory given")
            return
        names = self.get_model_names()
        fname = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in names:
            path = os.path.join(directory, name, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            ckpt = load_checkpoint(path)
            print(f'Loading {name} from "{path}" (epoch={ckpt["epoch"]})')
            self._models[name].load_state_dict(ckpt["state_dict"], strict=False)
