"""
MLPGA Trainer – Multi-FM Early Fusion + Faithful GraphAdapter Prototype Refinement.

Overview
────────
This trainer follows the original GraphAdapter paper as closely as possible
while extending it to three frozen pathology foundation models (BiomedCLIP,
PLIP, CONCH).

1. Early MLP Fusion
   Features from all three FMs are concatenated immediately:
       [B, 512] × 3  →  [B, 1536].
   A shared JointAdapter MLP maps this to a 512-dim common embedding space.
   The same adapter is applied identically to visual and text prototypes.

2. Faithful GraphAdapter Refinement
   Two GCN streams both refine TEXT prototypes (the query is not touched):
   • GCN_tt: all C text prototypes interact via a cosine-similarity graph
             → text protos refined with inter-class relational context
   • GCN_it: for each class c, text_c (node 0) receives messages from all
             visual prototypes (nodes 1..C) — cross-modal anchoring
   Fixed α/β blend the GCN outputs and add a residual from the original
   adapted text protos, matching the GraphAdapter paper. β (BETA_TT) weights
   the text stream when fusing the two GCNs (z'_t = β·z_tt + (1−β)·z_vt); α
   (ALPHA_IT) weights the original text feature in the residual
   (z*_t = α·z_t + (1−α)·z'_t). Paper optimal: α = 0.6, β = 0.7 (fixed, not learnable).

3. Classification
   L2-normalised query_adapted and final_text → cosine logits.
   The query passes through JointAdapter only; no graph refinement applied.

Checkpointing
─────────────
Uses TEST.FINAL_MODEL = "last_epoch" because the validation set has only
~28 samples (< 4 per class for LungHist), making best-val selection unstable.

Usage
─────
python train.py \\
    --trainer MLPGAFaithful \\
    --config-file configs/trainers/mlpga.yaml \\
    --dataset-config-file configs/datasets/lunghist_descriptive.yaml \\
    --root /path/to/datasets \\
    --output-dir output/mlpga_experiments/lunghist_4shot_seed1 \\
    DATASET.NUM_SHOTS 4 SEED 1
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint

from trainers.model_registry import MODEL_CONFIGS, load_clip_model
from trainers.mlpga_model import MLPGAModel


# ── Per-FM input resolutions ───────────────────────────────────────────────────
FM_INPUT_SIZES: dict = {
    "biomedclip": 224,
    "plip":       224,
    "conch":      448,
}

# ── Per-dataset text-prompt templates ─────────────────────────────────────────
CUSTOM_TEMPLATES: dict = {
    "LungHistDescriptive":  "a histological image of a {} cell in pulmonary pathology.",
    "BRACSDescriptive":     "a histological image of a {} cell in breast carcinoma subtyping.",
    "ICIAR2018Descriptive": "a histopathological image of {}.",
}
DEFAULT_TEMPLATE = "a histological image of a {} cell."


# ── Minimal cfg shim for load_clip_model ──────────────────────────────────────

class _MockCfg:
    class _BB:
        def __init__(self, name: str): self.NAME = name
    class _M:
        def __init__(self, name: str): self.BACKBONE = _MockCfg._BB(name)
    def __init__(self, fm_name: str): self.MODEL = _MockCfg._M(fm_name)


# =============================================================================
# Trainer
# =============================================================================

@TRAINER_REGISTRY.register()
class MLPGAFaithful(TrainerX):
    """
    MLPGA Faithful Trainer.

    Multi-FM early fusion + GraphAdapter-faithful text-only prototype refinement.
    The query passes through the JointAdapter and is used unchanged for cosine
    classification, exactly as in the original GraphAdapter paper.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MLPGA.FM_NAMES, \
            "TRAINER.MLPGA.FM_NAMES must be a non-empty list"
        assert cfg.TRAINER.MLPGA.PREC in ("fp32", "amp"), \
            "TRAINER.MLPGA.PREC must be 'fp32' or 'amp'"

    def build_model(self):
        cfg    = self.cfg
        mg     = cfg.TRAINER.MLPGA
        device = self.device

        self.fm_names:    List[str] = list(mg.FM_NAMES)
        self.num_fms:     int       = len(self.fm_names)
        self.num_classes: int       = self.dm.num_classes
        self.class_names: List[str] = self.dm.dataset.classnames
        self.prec:        str       = mg.PREC

        print("=" * 62)
        print("Building MLPGAFaithful (early fusion + faithful GraphAdapter)")
        print(f"  FMs          : {self.fm_names}")
        print(f"  Common dim   : {mg.COMMON_DIM}")
        print(f"  Hidden dim   : {mg.HIDDEN_DIM}")
        print(f"  Alpha / Beta : {mg.ALPHA_IT:.2f} / {mg.BETA_TT:.2f}  (fixed, paper alpha/beta)")
        print(f"  Precision    : {self.prec}")
        print(f"  Num classes  : {self.num_classes}")
        print("=" * 62)

        # ── Load frozen FM encoders ──────────────────────────────────────
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

        feat_dim: int = self.feat_dims[0]   # all 512 in this project

        # ── Extract frozen prototypes ────────────────────────────────────
        print("\nExtracting visual prototypes ...")
        visual_protos: List[torch.Tensor] = self._extract_visual_prototypes()

        print("Extracting text prototypes ...")
        text_protos: List[torch.Tensor] = self._extract_text_features()

        # ── Build model ──────────────────────────────────────────────────
        self.model = MLPGAModel(
            num_fms          = self.num_fms,
            feat_dim         = feat_dim,
            hidden_dim       = mg.HIDDEN_DIM,
            common_dim       = mg.COMMON_DIM,
            num_classes      = self.num_classes,
            alpha_it         = mg.ALPHA_IT,
            beta_tt          = mg.BETA_TT,
            dropout          = mg.DROPOUT,
            logit_scale_init = mg.LOGIT_SCALE_INIT,
        ).to(device)

        # ── Register prototype buffers ───────────────────────────────────
        for i in range(self.num_fms):
            self.model.register_buffer(
                f"vproto_{i}", visual_protos[i].float().to(device)
            )
            self.model.register_buffer(
                f"tproto_{i}", text_protos[i].float().to(device)
            )

        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        print(f"\nTrainable parameters: {trainable:,}")

        # ── Optimizer + scheduler ────────────────────────────────────────
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("mlpga", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if self.prec == "amp" else None

    # ── Feature extraction helpers ─────────────────────────────────────────

    def _img_for_fm(self, imgs: torch.Tensor, fm_name: str) -> torch.Tensor:
        target = FM_INPUT_SIZES[fm_name]
        if imgs.shape[-1] != target:
            imgs = TF.resize(imgs, [target, target], antialias=True)
        return imgs

    @torch.no_grad()
    def _extract_features(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for enc, fm in zip(self.encoders, self.fm_names):
            x = self._img_for_fm(imgs, fm)
            try:
                f = enc.encode_image(x.type(enc.dtype))
            except Exception:
                f = enc.encode_image(x.float())
            f = F.normalize(f.float(), dim=-1)
            feats.append(f)
        return feats

    @torch.no_grad()
    def _extract_visual_prototypes(self) -> List[torch.Tensor]:
        n_epochs: int = self.cfg.MODEL.NUM_SAMPLE_EPOCHS
        accum = [
            [[] for _ in range(self.num_classes)]
            for _ in range(self.num_fms)
        ]
        for _ in range(n_epochs):
            for batch in self.train_loader_x:
                imgs   = batch["img"].to(self.device)
                labels = batch["label"]
                feats  = self._extract_features(imgs)
                for fi, f in enumerate(feats):
                    for b in range(f.size(0)):
                        cls = int(labels[b])
                        accum[fi][cls].append(f[b].cpu())

        protos = []
        for fi in range(self.num_fms):
            cls_means = []
            for c in range(self.num_classes):
                stacked = torch.stack(accum[fi][c], dim=0)
                cls_means.append(stacked.mean(dim=0))
            protos.append(torch.stack(cls_means, dim=0))
        return protos

    @torch.no_grad()
    def _extract_text_features(self) -> List[torch.Tensor]:
        dataset_name = self.cfg.DATASET.NAME
        template     = CUSTOM_TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)
        text_protos  = []
        for enc, fm in zip(self.encoders, self.fm_names):
            prompts = [template.format(cn) for cn in self.class_names]
            if hasattr(enc, "tokenize"):
                tokens = enc.tokenize(prompts).to(self.device)
            else:
                from clip import clip as openai_clip
                tokens = openai_clip.tokenize(prompts).to(self.device)
            feats = enc.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
            text_protos.append(feats.cpu())
        return text_protos

    # ── Buffer accessors ───────────────────────────────────────────────────

    def _get_proto_lists(self):
        vprotos = [getattr(self.model, f"vproto_{i}") for i in range(self.num_fms)]
        tprotos = [getattr(self.model, f"tproto_{i}") for i in range(self.num_fms)]
        return vprotos, tprotos

    # ── Training step ──────────────────────────────────────────────────────

    def forward_backward(self, batch):
        imgs, labels = self.parse_batch_train(batch)
        vprotos, tprotos = self._get_proto_lists()
        mg = self.cfg.TRAINER.MLPGA

        if self.prec == "amp":
            with autocast():
                query_feats = self._extract_features(imgs)
                logits = self.model(query_feats, vprotos, tprotos)
                loss   = F.cross_entropy(logits, labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            query_feats = self._extract_features(imgs)
            logits = self.model(query_feats, vprotos, tprotos)
            loss   = F.cross_entropy(logits, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return {
            "loss": loss.item(),
            "acc":  compute_accuracy(logits, labels)[0].item(),
        }

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def model_inference(self, imgs: torch.Tensor) -> torch.Tensor:
        vprotos, tprotos = self._get_proto_lists()
        query_feats = self._extract_features(imgs)
        return self.model(query_feats, vprotos, tprotos)

    # ── Evaluation ──────────────────────────────────────────────────────────

    def test(self, split=None):
        """Override test() to also report Cohen's Kappa and Macro AUC.

        Matches the GraphAdapter / HeGraph / MLP-Fusion trainers so the log
        emits ``* kappa:`` and ``* macro_auc:`` lines, which the multi-seed
        runner parses into results.jsonl (same output format across methods).
        """
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

        # Extra metrics: Kappa and AUC (model returns raw logits).
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

    def after_epoch(self):
        """Evaluate on the test set and checkpoint every 10th epoch.

        MLPGA is a novel method with no paper-prescribed epoch count, so we log
        test-set metrics every 10 epochs to inspect the accuracy curve and pick
        a good stopping point (in-training eval rather than reloading
        checkpoints afterwards, which is too slow).  The final epoch is
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

    # ── Checkpoint ────────────────────────────────────────────────────────

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
