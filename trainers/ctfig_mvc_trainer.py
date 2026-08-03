"""
CrossFM Transductive Instance Graph — Multi-View Cosine (CTFIG-MVC) Trainer.

Combines two ideas:
  1. Multi-view nodes (from CTFIG-MV): each augmented view of a support image
     is kept as a separate graph node instead of averaged.
  2. Cosine classification head (replacing the MLP in CTFIG/CTFIG-MV): after
     graph refinement, pool support nodes per class, blend with text prototypes,
     classify via cosine similarity. Only 4 learnable scalars vs ~1.18M MLP params.

At 4-shot with NUM_SAMPLE_EPOCHS=10 the support graph has 280 nodes per FM
(4 shots × 10 views × 7 classes) vs 28 in CTFIG.

Usage
-----
python train.py \\
    --trainer CrossFMTransductiveInstanceGraphMVC \\
    --config-file configs/trainers/ctfig_mvc.yaml \\
    --dataset-config-file configs/datasets/lunghist_descriptive.yaml \\
    --root /path/to/datasets \\
    --output-dir output/ctfig_mvc_experiments/lunghist_4shot_seed11111 \\
    DATASET.NUM_SHOTS 4 SEED 11111
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint

from trainers.model_registry import MODEL_CONFIGS, load_clip_model
from trainers.ctfig_mvc_model import CrossFMTransductiveInstanceGraphMVCModel


FM_INPUT_SIZES: dict = {
    "biomedclip": 224,
    "plip":       224,
    "conch":      448,
}

CUSTOM_TEMPLATES: dict = {
    "LungHistDescriptive":  "a histological image of a {} cell in pulmonary pathology.",
    "BRACSDescriptive":     "a histological image of a {} cell in breast carcinoma subtyping.",
    "ICIAR2018Descriptive": "a histopathological image of {}.",
}
DEFAULT_TEMPLATE = "a histological image of a {} cell."


class _MockCfg:
    class _BB:
        def __init__(self, name): self.NAME = name
    class _M:
        def __init__(self, name): self.BACKBONE = _MockCfg._BB(name)
    def __init__(self, fm_name): self.MODEL = _MockCfg._M(fm_name)


@TRAINER_REGISTRY.register()
class CrossFMTransductiveInstanceGraphMVC(TrainerX):
    """
    CrossFM Transductive Instance Graph — Multi-View Cosine (CTFIG-MVC).

    Each augmented view of every support image is a separate graph node.
    After graph refinement, the query is classified by cosine similarity against
    blended visual+text prototypes — no MLP, just 4 learnable scalars.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CTFIG.FM_NAMES, \
            "TRAINER.CTFIG.FM_NAMES must be a non-empty list"
        assert cfg.TRAINER.CTFIG.PREC in ("fp32", "amp"), \
            "TRAINER.CTFIG.PREC must be 'fp32' or 'amp'"

    def build_model(self):
        cfg   = self.cfg
        ctfig = cfg.TRAINER.CTFIG
        device = self.device

        self.fm_names:    List[str] = list(ctfig.FM_NAMES)
        self.num_fms:     int       = len(self.fm_names)
        self.num_classes: int       = self.dm.num_classes
        self.class_names: List[str] = self.dm.dataset.classnames
        self.prec:        str       = ctfig.PREC

        print("=" * 66)
        print("Building CrossFMTransductiveInstanceGraphMVC (multi-view + cosine head)")
        print(f"  FMs              : {self.fm_names}")
        print(f"  Num classes      : {self.num_classes}")
        print(f"  GNN layers       : {ctfig.NUM_GNN_LAYERS}")
        print(f"  Top-K            : {ctfig.TOP_K}")
        print(f"  Intramodal GNN   : {ctfig.USE_INTRAMODAL_GNN}")
        print(f"  Crossmodal GNN   : {ctfig.USE_CROSSMODAL_GNN}")
        print(f"  Cross-FM GNN     : {ctfig.USE_CROSSFM_GNN}")
        print(f"  alpha_v init     : {ctfig.ALPHA_V_INIT}")
        print(f"  alpha_t init     : {ctfig.ALPHA_T_INIT}")
        print(f"  gate_vt init     : {ctfig.GATE_VT_INIT}")
        print(f"  logit_scale init : {ctfig.LOGIT_SCALE_INIT}")
        print(f"  Transductive     : {ctfig.USE_TRANSDUCTIVE}")
        print(f"  Share crossmodal W: {ctfig.SHARE_CROSSMODAL_W}")
        print(f"  CrossFM attention : {ctfig.CROSSFM_ATTENTION}")
        print(f"  Sample epochs    : {cfg.MODEL.NUM_SAMPLE_EPOCHS}  (each view = 1 node)")
        print(f"  Precision        : {self.prec}")
        print("=" * 66)

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

        feat_dim: int = self.feat_dims[0]

        print("\nExtracting multi-view support features ...")
        vsupport, self.num_shots = self._extract_visual_support()
        n_epochs = cfg.MODEL.NUM_SAMPLE_EPOCHS
        print(f"  Support nodes per FM: {vsupport[0].size(0)} "
              f"({self.num_classes} classes x {self.num_shots} shots x {n_epochs} views)")

        print("Extracting text prototypes ...")
        text_protos = self._extract_text_features()

        self.model = CrossFMTransductiveInstanceGraphMVCModel(
            num_fms          = self.num_fms,
            feat_dim         = feat_dim,
            num_classes      = self.num_classes,
            num_gnn_layers   = ctfig.NUM_GNN_LAYERS,
            top_k            = ctfig.TOP_K,
            dropout          = ctfig.DROPOUT,
            alpha_v_init     = ctfig.ALPHA_V_INIT,
            alpha_t_init     = ctfig.ALPHA_T_INIT,
            gate_vt_init     = ctfig.GATE_VT_INIT,
            logit_scale_init = ctfig.LOGIT_SCALE_INIT,
            gate_cv_init     = ctfig.GATE_CV_INIT,
            gate_ct_init     = ctfig.GATE_CT_INIT,
            gate_xv_init     = ctfig.GATE_XV_INIT,
            gate_xt_init     = ctfig.GATE_XT_INIT,
            use_intramodal   = ctfig.USE_INTRAMODAL_GNN,
            use_crossmodal   = ctfig.USE_CROSSMODAL_GNN,
            use_crossfm      = ctfig.USE_CROSSFM_GNN,
            use_transductive = ctfig.USE_TRANSDUCTIVE,
            share_crossmodal_w = ctfig.SHARE_CROSSMODAL_W,
            crossfm_attention  = ctfig.CROSSFM_ATTENTION,
            use_learned_fm_weights = ctfig.LEARNED_FM_WEIGHTS,
            use_prototype_nodes = ctfig.USE_PROTOTYPE_NODES,
        ).to(device)

        for i in range(self.num_fms):
            self.model.register_buffer(
                f"vsupport_{i}", vsupport[i].float().to(device)
            )
            self.model.register_buffer(
                f"tproto_{i}", text_protos[i].float().to(device)
            )

        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        print(f"\nTrainable parameters: {trainable:,}")

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("ctfig_mvc", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if self.prec == "amp" else None

        init_path = ctfig.INIT_WEIGHTS
        if init_path:
            ckpt = load_checkpoint(init_path)
            sd = ckpt["state_dict"]
            param_keys = {n for n, _ in self.model.named_parameters()}
            cross_fm_patterns = {"W_xV", "W_xT", "gate_xv", "gate_xt"}
            filtered = {
                k: v for k, v in sd.items()
                if k in param_keys and not any(p in k for p in cross_fm_patterns)
            }
            skipped_xfm = [k for k in sd if any(p in k for p in cross_fm_patterns)]
            missing, _ = self.model.load_state_dict(filtered, strict=False)
            not_loaded = [k for k in (missing or []) if k in param_keys]
            print(f"\nWarm-start: loaded {len(filtered)} params from {init_path}")
            if skipped_xfm:
                print(f"  Skipped cross-FM params (fresh init): {skipped_xfm}")
            if not_loaded:
                print(f"  Not initialized (fresh): {not_loaded}")

    # ── Feature extraction ─────────────────────────────────────────────────

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
            feats.append(F.normalize(f.float(), dim=-1))
        return feats

    def _make_support_loader(self):
        """
        Dedicated DataLoader for support feature extraction.

        Uses drop_last=False and shuffle=False so that every support image is
        seen exactly once per pass, guaranteeing C*N images total (equal per class)
        and exactly n_epochs views per image. The training DataLoader uses
        drop_last=True which would drop images at 16-shot (112 images, batch 32).
        """
        from torch.utils.data import DataLoader
        train_loader = self.train_loader_x
        return DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            collate_fn=train_loader.collate_fn,
        )

    @torch.no_grad()
    def _extract_visual_support(self) -> Tuple[List[torch.Tensor], int]:
        """
        Extract support features keeping every augmented view as a separate node.

        Uses a drop_last=False loader so all C*N support images are seen every
        epoch, giving exactly n_epochs views per image and equal class sizes.

        Returns
        -------
        vsupport  : list[K] of [C*N*n_epochs, feat_dim]  sorted by (class, impath)
        num_shots : int  shots per class (N)
        """
        n_epochs    = self.cfg.MODEL.NUM_SAMPLE_EPOCHS
        support_loader = self._make_support_loader()

        accum: List[Dict[str, List[torch.Tensor]]] = [
            defaultdict(list) for _ in range(self.num_fms)
        ]
        impath_to_label: Dict[str, int] = {}

        for _ in range(n_epochs):
            for batch in support_loader:
                imgs    = batch["img"].to(self.device)
                labels  = batch["label"]
                impaths = batch["impath"]
                feats   = self._extract_features(imgs)

                for b in range(imgs.size(0)):
                    ip  = impaths[b]
                    lbl = int(labels[b])
                    impath_to_label[ip] = lbl
                    for fi in range(self.num_fms):
                        accum[fi][ip].append(feats[fi][b].cpu())

        # All images seen every epoch → sort by (class, path) and stack views.
        # The few-shot split guarantees N images per class, so the total
        # C*N*n_epochs is always divisible by C without any truncation.
        sorted_impaths = sorted(
            impath_to_label.keys(),
            key=lambda p: (impath_to_label[p], p)
        )

        vsupport = []
        for fi in range(self.num_fms):
            node_feats = torch.cat(
                [torch.stack(accum[fi][ip], dim=0) for ip in sorted_impaths],
                dim=0,
            )
            vsupport.append(node_feats)

        num_shots = len(sorted_impaths) // self.num_classes
        return vsupport, num_shots

    @torch.no_grad()
    def _extract_text_features(self) -> List[torch.Tensor]:
        dataset_name = self.cfg.DATASET.NAME
        template     = CUSTOM_TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)

        text_protos = []
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

    def _get_support_and_proto_lists(self):
        vsupport = [getattr(self.model, f"vsupport_{i}") for i in range(self.num_fms)]
        tprotos  = [getattr(self.model, f"tproto_{i}")   for i in range(self.num_fms)]
        return vsupport, tprotos

    # ── Training step ──────────────────────────────────────────────────────

    def forward_backward(self, batch):
        imgs, labels = self.parse_batch_train(batch)
        vsupport, tprotos = self._get_support_and_proto_lists()
        ctfig = self.cfg.TRAINER.CTFIG

        if self.prec == "amp":
            with autocast():
                query_feats = self._extract_features(imgs)
                logits      = self.model(query_feats, vsupport, tprotos)
                loss        = F.cross_entropy(logits, labels,
                                              label_smoothing=ctfig.LABEL_SMOOTHING)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            query_feats = self._extract_features(imgs)
            logits      = self.model(query_feats, vsupport, tprotos)
            loss        = F.cross_entropy(logits, labels,
                                          label_smoothing=ctfig.LABEL_SMOOTHING)
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

    @torch.no_grad()
    def model_inference(self, imgs: torch.Tensor) -> torch.Tensor:
        vsupport, tprotos = self._get_support_and_proto_lists()
        query_feats = self._extract_features(imgs)
        return self.model(query_feats, vsupport, tprotos)

    def test(self, split=None):
        """Evaluate and report accuracy, macro-F1, Cohen's Kappa and Macro AUC.

        Mirrors the GraphAdapter / HeGraph / MLPGA trainers so the log emits
        ``* kappa:`` and ``* macro_auc:`` lines, which the multi-seed runner
        (run_utils) parses into results.jsonl -- identical output format across
        methods.  Also appends the per-epoch curve to test_per_epoch.csv.
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

        # Per-epoch curve (the metrics also land in results.jsonl via the marker
        # printed in after_epoch, but the CSV keeps a self-contained record).
        csv_path = os.path.join(self.output_dir, "test_per_epoch.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("epoch,accuracy,macro_f1,kappa,macro_auc\n")
            f.write(f"{self.epoch + 1},{results['accuracy']:.2f},"
                    f"{results['macro_f1']:.2f},{kappa:.4f},{auc:.4f}\n")

        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)

        return list(results.values())[0]

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        freq = self.cfg.TRAIN.CHECKPOINT_FREQ
        meet_checkpoint_freq = (
            (self.epoch + 1) % freq == 0 if freq > 0 else False
        )

        # Checkpoint every CHECKPOINT_FREQ epochs (and at the end) so a specific
        # epoch can be reloaded later.
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        # Intermediate test evaluation every CHECKPOINT_FREQ epochs.  The last
        # epoch is evaluated by after_train(), so it is skipped here.  The marker
        # line lets run_utils.parse_log_metrics_per_epoch record one jsonl row
        # per evaluated epoch -- matching MLPGA / MLPHeGraph.
        if meet_checkpoint_freq and not last_epoch:
            print(f"\n=> Intermediate test evaluation at epoch {self.epoch + 1}")
            self.test(split="test")
            self.set_model_mode("train")

    def load_model(self, directory, epoch=None):
        if not directory:
            print("load_model() skipped -- no directory given")
            return
        names = self.get_model_names()
        fname = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        for name in names:
            path = os.path.join(directory, name, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            ckpt = load_checkpoint(path)
            print(f'Loading {name} from "{path}" (epoch={ckpt["epoch"]})')
            missing, unexpected = self._models[name].load_state_dict(
                ckpt["state_dict"], strict=False
            )
            if missing:
                print(f"  Warning: missing keys   : {missing}")
            if unexpected:
                print(f"  Warning: unexpected keys: {unexpected}")
