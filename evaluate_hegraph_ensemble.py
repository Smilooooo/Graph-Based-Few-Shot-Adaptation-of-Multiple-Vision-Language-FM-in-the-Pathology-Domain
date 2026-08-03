"""
HeGraphAdapter Ensemble — Majority-Vote Evaluation from Single-Backbone Runs.

Reuses trained models from output/hegraph_experiments/ (produced by
run_hegraph_multiseed.py).  For each (dataset, shots, seed) combination where
all 3 backbones completed, loads each checkpoint, runs inference on the test
set, and computes the majority-vote ensemble.

Mirrors evaluate_graphadapter_ensemble.py exactly (same discovery, resume,
JSONL output, and metrics) so GraphAdapter and HeGraph ensembles are directly
comparable.  Differences: HeGraph trainer/configs, epoch-30 checkpoints, and the
model returns fused softmax probabilities in eval mode (not raw logits).

Metrics per backbone and per ensemble: accuracy, macro_f1, weighted_f1, kappa,
macro_auc.

Usage
-----
python evaluate_hegraph_ensemble.py
python evaluate_hegraph_ensemble.py --datasets lunghist --shots 4
python evaluate_hegraph_ensemble.py --experiments-dir output/hegraph_experiments
"""

import argparse
import gc
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from tqdm import tqdm

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed

# Register custom datasets and trainer
import datasets.lungHist700_tien_descriptive
import datasets.bracs_tien_descriptive
import datasets.iciar2018_tien_descriptive
import trainers.he_graph_paper

# ── Config maps (must match run_hegraph_multiseed.py) ─────────────────────────

DATASET_CONFIGS = {
    "lunghist": "configs/datasets/lunghist_descriptive.yaml",
    "bracs":    "configs/datasets/bracs_descriptive.yaml",
    "iciar":    "configs/datasets/iciar2018_descriptive.yaml",
}

BACKBONE_CONFIGS = {
    "biomedclip": "configs/trainers/biomedclip.yaml",
    "plip":       "configs/trainers/plip.yaml",
    "conch":      "configs/trainers/conch.yaml",
}

BACKBONES = ["biomedclip", "plip", "conch"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extend_cfg(cfg):
    """Mirror extend_cfg from train_hegraph.py / ensemble_train_eval.py."""
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16
    cfg.TRAINER.COOP.CSC = False
    cfg.TRAINER.COOP.CTX_INIT = ""
    cfg.TRAINER.COOP.PREC = "fp16"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""
    cfg.TRAINER.COCOOP.PREC = "fp16"

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

    cfg.MODEL.NUM_SAMPLE_EPOCHS = 10
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"


def _extra_metrics(y_true, y_pred, y_prob):
    num_classes = y_prob.shape[1]
    m = {}
    m["macro_f1"]    = f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    m["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    m["kappa"]       = cohen_kappa_score(y_true, y_pred)
    try:
        mc = "ovr" if num_classes > 2 else "raise"
        m["macro_auc"] = roc_auc_score(y_true, y_prob, multi_class=mc, average="macro")
    except Exception:
        m["macro_auc"] = float("nan")
    return m


def majority_vote(predictions_dict, labels):
    backbone_names = list(predictions_dict.keys())
    n_samples = labels.shape[0]
    all_preds = torch.stack([predictions_dict[b] for b in backbone_names])

    ensemble_preds = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        votes = all_preds[:, i].tolist()
        ensemble_preds[i] = Counter(votes).most_common(1)[0][0]

    accuracy = (ensemble_preds == labels).float().mean().item() * 100.0
    return ensemble_preds, accuracy


# ── Discover completed runs ──────────────────────────────────────────────────

def discover_runs(experiments_dir, datasets, shots, seeds):
    """Group completed single-backbone runs by (dataset, shots, seed).

    Returns {(dataset, shots, seed): {backbone: model_dir}} only where
    all 3 backbones finished training.
    """
    from run_utils import is_run_complete

    groups = defaultdict(dict)

    for dataset in datasets:
        for s in shots:
            for seed in seeds:
                for bb in BACKBONES:
                    run_dir = os.path.join(
                        experiments_dir,
                        f"{dataset}_{bb}_{s}shot_seed{seed}",
                    )
                    if is_run_complete(run_dir):
                        groups[(dataset, s, seed)][bb] = run_dir

    complete = {k: v for k, v in groups.items() if len(v) == len(BACKBONES)}
    return complete


# ── Load & evaluate one backbone ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_backbone(model_dir, dataset_cfg, backbone_cfg, num_shots, seed, root):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(dataset_cfg)
    cfg.merge_from_file(backbone_cfg)
    cfg.DATASET.ROOT = root
    cfg.DATASET.NUM_SHOTS = num_shots
    cfg.SEED = seed
    cfg.OUTPUT_DIR = model_dir
    cfg.TRAINER.NAME = "HeGraphCLIPPaper"
    cfg.freeze()

    set_random_seed(cfg.SEED)

    trainer = build_trainer(cfg)
    trainer.load_model(model_dir, epoch=cfg.OPTIM.MAX_EPOCH)
    trainer.set_model_mode("eval")   # sets alpha_n2p to its test-time value (0.1)

    all_preds, all_probs, all_labels = [], [], []
    for batch in tqdm(trainer.test_loader, desc="    inference", leave=False):
        images = batch["img"].cuda()
        labels = batch["label"].cuda()
        # HeGraph returns fused softmax probabilities in eval mode (is_training=False)
        output = trainer.model(images)
        all_preds.append(output.argmax(dim=1).cpu())
        all_probs.append(output.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)
    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item() * 100
    probs = all_probs.numpy()   # already softmax — do NOT re-normalise
    metrics = _extra_metrics(all_labels.numpy(), all_preds.numpy(), probs)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "preds": all_preds,
        "labels": all_labels,
        "probs": probs,
        "accuracy": acc,
        **metrics,
    }


# ── JSONL helpers (resume + results) ─────────────────────────────────────────

def load_completed_from_jsonl(jsonl_path):
    """Load already-evaluated ensembles from the JSONL file.

    Returns {(dataset, shots, seed): record_dict}.
    """
    done = {}
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARNING: skipping malformed JSONL line in {jsonl_path}")
                continue
            key = (r["dataset"], r["shots"], r["seed"])
            done[key] = r
    return done


def append_jsonl(jsonl_path, record):
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HeGraphAdapter ensemble (majority vote) from single-backbone runs")
    parser.add_argument("--experiments-dir", type=str,
                        default="output/hegraph_experiments")
    parser.add_argument("--output-dir", type=str,
                        default="output/hegraph_ensemble")
    parser.add_argument("--root", type=str,
                        default=r"E:\BachelorThesis\Data\data(3)\data")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DATASET_CONFIGS),
                        default=list(DATASET_CONFIGS))
    parser.add_argument("--shots", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[11111, 22222, 33333, 44444, 55555])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "results.jsonl")

    groups = discover_runs(args.experiments_dir, args.datasets, args.shots, args.seeds)
    if not groups:
        print("No complete (all 3 backbones) run groups found.")
        return

    done = load_completed_from_jsonl(jsonl_path)

    print(f"Found {len(groups)} complete (dataset, shots, seed) groups.\n")

    all_results = []
    total = len(groups)
    skipped = 0

    for idx, ((dataset, shots, seed), backbone_dirs) in enumerate(sorted(groups.items()), 1):
        key = (dataset, shots, seed)

        if key in done:
            skipped += 1
            mv = done[key]["majority_vote"]
            print(f"[{idx}/{total}] {dataset} {shots}-shot seed={seed}  "
                  f"SKIPPED (MV={mv:.2f}%)")
            all_results.append(done[key])
            continue

        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] {dataset} {shots}-shot seed={seed}")
        print(f"{'='*60}")

        dataset_cfg = DATASET_CONFIGS[dataset]
        results_by_bb = {}

        for bb in BACKBONES:
            model_dir = backbone_dirs[bb]
            backbone_cfg = BACKBONE_CONFIGS[bb]
            print(f"  Evaluating {bb}...", end=" ", flush=True)
            try:
                result = evaluate_backbone(
                    model_dir, dataset_cfg, backbone_cfg, shots, seed, args.root)
                results_by_bb[bb] = result
                print(f"Acc: {result['accuracy']:.2f}%  "
                      f"F1: {result['macro_f1']:.2f}%  "
                      f"Kappa: {result['kappa']:.4f}  "
                      f"AUC: {result['macro_auc']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")

        if len(results_by_bb) < len(BACKBONES):
            print("  Skipping ensemble (not all backbones evaluated)")
            continue

        # Verify all backbones saw the test set in the same order before voting
        labels = results_by_bb[BACKBONES[0]]["labels"]
        mismatch = next(
            (bb for bb in BACKBONES[1:]
             if not torch.equal(results_by_bb[bb]["labels"], labels)),
            None,
        )
        if mismatch is not None:
            print(f"  ERROR: test-set label order differs for {mismatch}; "
                  f"skipping ensemble for this group")
            continue

        # Majority vote
        predictions = {bb: results_by_bb[bb]["preds"] for bb in BACKBONES}
        mv_preds, mv_acc = majority_vote(predictions, labels)

        avg_probs = np.mean(
            [results_by_bb[bb]["probs"] for bb in BACKBONES], axis=0)
        mv_metrics = _extra_metrics(labels.numpy(), mv_preds.numpy(), avg_probs)

        print(f"  Majority Voting  : {mv_acc:.2f}%  "
              f"F1: {mv_metrics['macro_f1']:.2f}%  "
              f"Kappa: {mv_metrics['kappa']:.4f}  "
              f"AUC: {mv_metrics['macro_auc']:.4f}")

        record = {
            "method": "HeGraph_Ensemble",
            "dataset": dataset,
            "shots": shots,
            "seed": seed,
            "individual": {bb: results_by_bb[bb]["accuracy"] for bb in BACKBONES},
            "individual_metrics": {
                bb: {k: results_by_bb[bb][k]
                     for k in ("macro_f1", "weighted_f1", "kappa", "macro_auc")}
                for bb in BACKBONES
            },
            "majority_vote": mv_acc,
            "mv_metrics": mv_metrics,
        }
        append_jsonl(jsonl_path, record)
        all_results.append(record)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({skipped} skipped / {total} total)")
    print(f"{'='*60}")

    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["shots"])].append(r)

    for (dataset, shots), entries in sorted(grouped.items()):
        print(f"\n{dataset.upper()} {shots}-shot ({len(entries)} seeds):")
        for bb in BACKBONES:
            accs = [e["individual"][bb] for e in entries if bb in e.get("individual", {})]
            if accs:
                print(f"  {bb:15s}: {np.mean(accs):.2f} +/- {np.std(accs):.2f}%")
        mv_accs = [e["majority_vote"] for e in entries]
        print(f"  {'MV Ensemble':15s}: {np.mean(mv_accs):.2f} +/- {np.std(mv_accs):.2f}%")

    print(f"\nResults: {jsonl_path}")


if __name__ == "__main__":
    main()
