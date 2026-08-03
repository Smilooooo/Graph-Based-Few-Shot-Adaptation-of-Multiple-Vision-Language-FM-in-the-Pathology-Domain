"""
MLP Fusion Baseline — Simple 3-layer MLP on concatenated features from all 3 backbones.

This is FEATURE FUSION, not an ensemble:
- Ensemble: multiple models → multiple predictions → vote/average
- Fusion: multiple models → concatenate features → single classifier

Features are extracted ON-THE-FLY from augmented images (like HeGraph does).
This ensures fair comparison with HeGraph/GraphAdapter.

Architecture:
- Input: Concatenation of BiomedCLIP, PLIP, CONCH features → 512*3 = 1536 dims
- Layer 1: 1536 → 512 (ReLU)
- Layer 2: 512 → 171 (ReLU)  
- Layer 3: 171 → num_classes (logits)

Parameter calculation (for 7 classes):
- Layer 1: (1536 × 512) + 512 bias = 786,944 params
- Layer 2: (512 × 171) + 171 bias = 87,723 params  
- Layer 3: (171 × 7) + 7 bias = 1,204 params
- Total: ~876k params (comparable to HeGraph ~750k, GraphAdapter ~500k)

The CLIP encoders are frozen; only the MLP is trained.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_checkpoint
from dassl.metrics import compute_accuracy
from torch.cuda.amp import GradScaler, autocast

from trainers.model_registry import load_clip_model, MODEL_CONFIGS

BACKBONES = ["biomedclip", "plip", "conch"]

# Per-FM expected input resolutions. Images come from the dataloader at the
# max resolution (448 for CONCH) and are downsampled per backbone as needed.
FM_INPUT_SIZES = {
    "biomedclip": 224,
    "plip": 224,
    "conch": 448,
}


# ---------------------------------------------------------------------------
# 3-Layer MLP Classifier
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """3-layer MLP where each layer is 1/3 of the previous.
    
    1536 → 512 → 171 → num_classes (~876k params)
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        hidden1 = input_dim // 3   # 512
        hidden2 = hidden1 // 3     # 171
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Mock config objects for loading models
# ---------------------------------------------------------------------------

class _MockBackboneConfig:
    """Mock config object for load_clip_model that mimics cfg.MODEL.BACKBONE."""
    def __init__(self, name):
        self.NAME = name


class _MockModelConfig:
    """Mock config object for load_clip_model that mimics cfg.MODEL."""
    def __init__(self, backbone_name):
        self.BACKBONE = _MockBackboneConfig(backbone_name)


class _MockConfig:
    """Mock config object for load_clip_model."""
    def __init__(self, backbone_name):
        self.MODEL = _MockModelConfig(backbone_name)


# ---------------------------------------------------------------------------
# Trainer (Dassl) with on-the-fly feature extraction
# ---------------------------------------------------------------------------

@TRAINER_REGISTRY.register()
class MLPFusionBaseline(TrainerX):
    """Trainer for MLP fusion baseline with on-the-fly feature extraction."""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        num_classes = len(classnames)
        
        # Get dropout from config or use default
        hegraph_cfg = getattr(cfg.TRAINER, 'HEGRAPH', None)
        dropout = getattr(hegraph_cfg, 'MLP_DROPOUT', 0.1) if hegraph_cfg else 0.1
        
        print(f"Building MLP Fusion Baseline (with on-the-fly feature extraction)")
        print(f"  Num classes: {num_classes}")
        print(f"  Dropout: {dropout}")
        
        # ----- Load encoders (keep them for on-the-fly extraction) -----
        self.encoders = self._load_encoders()
        
        # ----- Build MLP classifier -----
        total_feature_dim = sum(MODEL_CONFIGS[b]["feature_dim"] for b in BACKBONES)
        
        self.model = MLPClassifier(
            input_dim=total_feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.model.to(self.device)
        
        if cfg.TRAINER.COOP.PREC == "fp32":
            self.model.float()
        
        # Count params
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable parameters: {num_params:,}")
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("mlp_classifier", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
    
    def _load_encoders(self):
        """Load all 3 CLIP backbones."""
        encoders = {}
        print("Loading all 3 backbones for on-the-fly feature extraction...")
        
        for backbone_name in BACKBONES:
            print(f"  Loading {backbone_name}...")
            mock_cfg = _MockConfig(backbone_name)
            encoder = load_clip_model(mock_cfg, device="cpu")
            encoder.to(self.device)
            encoder.eval()
            
            for param in encoder.parameters():
                param.requires_grad_(False)
            
            encoders[backbone_name] = encoder
        
        return encoders
    
    @torch.no_grad()
    def _extract_features_from_images(self, images):
        """Extract concatenated features from all backbones for a batch of images.

        Images arrive at the dataloader's max resolution (448 to fit CONCH)
        and are resized per-backbone to that backbone's expected input size.
        """
        batch_features = []
        for backbone_name in BACKBONES:
            encoder = self.encoders[backbone_name]
            target = FM_INPUT_SIZES[backbone_name]
            x = images
            if x.shape[-1] != target or x.shape[-2] != target:
                x = TF.resize(x, [target, target], antialias=True)
            try:
                feat = encoder.encode_image(x.type(encoder.dtype))
            except:
                feat = encoder.encode_image(x.float())
            feat = F.normalize(feat, dim=-1)
            batch_features.append(feat.float())
        
        # Concatenate features from all backbones
        concat_features = torch.cat(batch_features, dim=-1)  # [B, 1536]
        return concat_features
    
    def forward_backward(self, batch):
        """Forward/backward on extracted features from images."""
        image, label, _ = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        
        # Extract features on-the-fly
        features = self._extract_features_from_images(image)
        
        if prec == "amp":
            with autocast():
                logits = self.model(features)
                loss = F.cross_entropy(logits, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits = self.model(features)
            loss = F.cross_entropy(logits, label)
            self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def model_inference(self, image):
        """Extract features and run MLP inference."""
        features = self._extract_features_from_images(image)
        return self.model(features)
    
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

    def after_epoch(self):
        """Evaluate on the test set and checkpoint every 10th epoch.

        MLP Fusion has no paper-prescribed number of epochs, so we log the
        test-set metrics every 10 epochs to inspect the accuracy curve and
        choose a good stopping point.  In-training evaluation is used instead
        of reloading checkpoints afterwards, which is too slow here.  The final
        epoch is evaluated by ``after_train()``, so it is skipped below.  A
        checkpoint (only the small trainable MLP head is registered) is also
        saved every 10 epochs so a specific epoch can be reloaded later.
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
            
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
