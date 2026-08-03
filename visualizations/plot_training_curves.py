"""
Training curves: test accuracy vs training epoch for the four multi-backbone methods.

Uses ONLY this repository's current, table-source run outputs (the same dirs verify_thesis_tables.py
reads, i.e. the post-tanh/clamp runs behind the thesis Ch.5 tables) — NOT the stale, deleted
*_epoch_sweep_* runs. Per-method sources, reusing the proven loaders so the epoch mapping matches the
thesis exactly:

  Cross-FM Instance Graph : output/ctfig_mvc_experiments/<run>/test_per_epoch.csv   (epochs 10..100)
  MLPHeGraphAdapter       : output/mlphegraph_experiments/results.jsonl             (intermediate evals)
  MLP Fusion              : output/mlp_fusion_experiments/results.jsonl             (60-epoch schedule)
  MLPGraphAdapter         : parsed from each run's log.txt (jsonl lost the per-epoch rows)

Note: only *intermediate* in-training evals are recorded, so a method's curve ends one eval-interval
before its final epoch (e.g. MLPHeGraph at 90, MLP Fusion at 50); the final-epoch point is stored
separately and is not included here. MLP Fusion uses a shorter 60-epoch schedule by design (not retrained
to 100). 3x3 grid: rows = datasets, columns = shots. Mean over 5 seeds, +/- 1 std bands, 30-epoch budget
marked.

Usage:  python visualizations/plot_training_curves.py
ASCII-only console output.
"""

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "visualizations")
sys.path.insert(0, PROJECT_ROOT)

DATASETS = [("lunghist", "LungHist700"), ("bracs", "BRACS"), ("iciar", "ICIAR2018")]
SHOTS = [4, 8, 16]
BUDGET_EPOCH = 30

METHODS = [
    ("MLP Fusion", "#1f77b4", "o"),
    ("MLPGraphAdapter", "#ff7f0e", "s"),
    ("MLPHeGraphAdapter", "#2ca02c", "^"),
    ("Cross-FM Instance Graph", "#d62728", "D"),
]


def _run_key(dirname):
    # e.g. bracs_16shot_seed11111
    try:
        ds = dirname.split("_")[0]
        shot = int(dirname.split("_")[1].replace("shot", ""))
        seed = int(dirname.split("seed")[1])
        return ds, shot, seed
    except (IndexError, ValueError):
        return None


def load_ctfig():
    """(ds, shot, epoch) -> [acc over seeds] from test_per_epoch.csv."""
    acc = defaultdict(list)
    for d in glob.glob(os.path.join(PROJECT_ROOT, "output", "ctfig_mvc_experiments", "*shot_seed*")):
        key = _run_key(os.path.basename(d))
        if key is None:
            continue
        ds, shot, _ = key
        p = os.path.join(d, "test_per_epoch.csv")
        if not os.path.isfile(p):
            continue
        for r in csv.DictReader(open(p)):
            acc[(ds, shot, int(r["epoch"]))].append(float(r["accuracy"]))
    return acc


def load_jsonl(rel):
    """(ds, shot, epoch) -> [acc] from an experiment-level results.jsonl with per-epoch rows."""
    acc = defaultdict(list)
    path = os.path.join(PROJECT_ROOT, rel)
    if not os.path.isfile(path):
        print("  WARNING missing %s" % rel)
        return acc
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("accuracy") is None or r.get("epoch") is None:
            continue
        acc[(r["dataset"], int(r["shots"]), int(r["epoch"]))].append(float(r["accuracy"]))
    return acc


def load_mlpga():
    """(ds, shot, epoch) -> [acc] parsed from each run's log.txt (reuses run_utils)."""
    from run_utils import parse_log_metrics_per_epoch
    acc = defaultdict(list)
    for d in glob.glob(os.path.join(PROJECT_ROOT, "output", "mlpga_experiments", "*shot_seed*")):
        key = _run_key(os.path.basename(d))
        if key is None:
            continue
        ds, shot, _ = key
        for row in parse_log_metrics_per_epoch(d) or []:
            if row.get("epoch") is not None and row.get("accuracy") is not None:
                acc[(ds, shot, int(row["epoch"]))].append(float(row["accuracy"]))
    return acc


# Full training schedule per method (final-epoch point to append; only intermediate evals are per-epoch).
SCHEDULE = {"MLP Fusion": 60, "MLPGraphAdapter": 100, "MLPHeGraphAdapter": 100}


def load_final(base, schedule_epoch):
    """(ds, shot, schedule_epoch) -> [acc]: each run's FINAL-epoch accuracy (last eval in the log)."""
    from run_utils import parse_log_metrics
    acc = defaultdict(list)
    for d in glob.glob(os.path.join(PROJECT_ROOT, "output", base, "*shot_seed*")):
        key = _run_key(os.path.basename(d))
        if key is None:
            continue
        ds, shot, _ = key
        m = parse_log_metrics(d)
        if m.get("accuracy") is not None:
            acc[(ds, shot, schedule_epoch)].append(float(m["accuracy"]))
    return acc


def load_hbaseline(base, backbone="conch"):
    """(ds, shot) -> mean final accuracy of a single-backbone baseline (horizontal reference line)."""
    path = os.path.join(PROJECT_ROOT, "output", base, "results.jsonl")
    if not os.path.isfile(path):
        print("  WARNING missing %s" % path)
        return {}
    perkey = {}  # (ds, shot, seed) -> acc  (dedup reruns, keep last, as verify_thesis_tables does)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("backbone") != backbone or r.get("accuracy") is None:
            continue
        perkey[(r["dataset"], int(r["shots"]), int(r["seed"]))] = float(r["accuracy"])
    acc = defaultdict(list)
    for (ds, shot, _), a in perkey.items():
        acc[(ds, shot)].append(a)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def curve(acc, ds, shot):
    """Return sorted epochs, mean, std for one (ds, shot)."""
    pts = sorted((ep, vals) for (d, s, ep), vals in acc.items() if d == ds and s == shot)
    eps = [ep for ep, _ in pts]
    means = np.array([np.mean(v) for _, v in pts])
    stds = np.array([np.std(v) for _, v in pts])
    return np.array(eps), means, stds


def plot_dataset_row(data, hbaselines, ds_key, ds_name):
    """One figure per dataset: a 1x3 row across all shots (complete, readable size)."""
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    lo, hi = 100.0, 0.0
    for ci, shot in enumerate(SHOTS):
        ax = axes[ci]
        for name, color, marker in METHODS:
            eps, means, stds = curve(data[name], ds_key, shot)
            if len(eps) == 0:
                continue
            ax.plot(eps, means, color=color, marker=marker, linewidth=1.9,
                    markersize=5, label=name)
            ax.fill_between(eps, means - stds, means + stds, color=color, alpha=0.10)
            lo = min(lo, float((means - stds).min()))
            hi = max(hi, float((means + stds).max()))
        for hname, hcolor, hvals in hbaselines:
            if (ds_key, shot) in hvals:
                y = hvals[(ds_key, shot)]
                ax.axhline(y, color=hcolor, linestyle="--", linewidth=1.4, alpha=0.85,
                           label=hname if ci == 0 else None)
                lo = min(lo, y); hi = max(hi, y)
        ax.axvline(BUDGET_EPOCH, color="grey", linestyle=":", linewidth=1.4, alpha=0.8,
                   label="30 epochs" if ci == 0 else None)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(list(range(10, 110, 20)))
        ax.set_title(f"{shot}-shot")
        ax.set_xlabel("Epoch")
        if ci == 0:
            ax.set_ylabel("Accuracy (%)")
    lo = max(0, np.floor(lo) - 2); hi = min(100, np.ceil(hi) + 2)
    for ax in axes:
        ax.set_ylim(lo, hi)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"training_curves_{ds_key}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("Saved: %s" % p)
    plt.close(fig)


def main():
    data = {
        "Cross-FM Instance Graph": load_ctfig(),
        "MLPHeGraphAdapter": load_jsonl("output/mlphegraph_experiments/results.jsonl"),
        "MLP Fusion": load_jsonl("output/mlp_fusion_experiments/results.jsonl"),
        "MLPGraphAdapter": load_mlpga(),
    }

    # append the final-epoch point (only intermediate evals were stored per-epoch)
    final_src = {"MLP Fusion": "mlp_fusion_experiments",
                 "MLPGraphAdapter": "mlpga_experiments",
                 "MLPHeGraphAdapter": "mlphegraph_experiments"}
    for name, base in final_src.items():
        for k, v in load_final(base, SCHEDULE[name]).items():
            data[name].setdefault(k, v)

    # single-backbone CONCH baselines -> horizontal reference lines
    # (reported at their paper schedules: GraphAdapter epoch 100, HeGraphAdapter epoch 30)
    hbaselines = [
        ("GraphAdapter (CONCH)", "#9467bd", load_hbaseline("graphadapter_experiments")),
        ("HeGraphAdapter (CONCH)", "#8c564b", load_hbaseline("hegraph_experiments")),
    ]

    # report coverage
    print("Epoch coverage per method:")
    for name, _, _ in METHODS:
        eps = sorted(set(ep for (_, _, ep) in data[name]))
        print("  %-26s epochs %s" % (name, "%d..%d" % (eps[0], eps[-1]) if eps else "NONE"))

    # Combined 3x3 grid (rows = datasets, columns = shots). Built near its final PRINTED
    # size so LaTeX barely rescales it -> readable fonts (a big figure gets shrunk to
    # \linewidth, which is what made the fonts look tiny before).
    # Original chart size (14x11); only the fonts are enlarged relative to it.
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16})
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))

    for ri, (ds_key, ds_name) in enumerate(DATASETS):
        lo, hi = 100.0, 0.0
        for ci, shot in enumerate(SHOTS):
            ax = axes[ri][ci]
            for name, color, marker in METHODS:
                eps, means, stds = curve(data[name], ds_key, shot)
                if len(eps) == 0:
                    continue
                ax.plot(eps, means, color=color, marker=marker, linewidth=1.6,
                        markersize=4, label=name)
                ax.fill_between(eps, means - stds, means + stds, color=color, alpha=0.10)
                lo = min(lo, float((means - stds).min()))
                hi = max(hi, float((means + stds).max()))
            for hname, hcolor, hvals in hbaselines:
                if (ds_key, shot) in hvals:
                    y = hvals[(ds_key, shot)]
                    ax.axhline(y, color=hcolor, linestyle="--", linewidth=1.3, alpha=0.85,
                               label=hname if (ri == 0 and ci == 0) else None)
                    lo = min(lo, y)
                    hi = max(hi, y)
            ax.axvline(BUDGET_EPOCH, color="grey", linestyle=":", linewidth=1.2, alpha=0.8,
                       label="30 epochs")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(list(range(10, 110, 20)))
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            if ri == 0:
                ax.set_title(f"{shot}-shot", fontsize=16)
            if ri == 2:
                ax.set_xlabel("Epoch")
            if ci == 0:
                ax.set_ylabel("Accuracy (%)")
            if ci == 2:
                ax2 = ax.twinx()
                ax2.set_ylabel(ds_name, fontsize=16, fontweight="bold", rotation=270, labelpad=18)
                ax2.set_yticks([])
        lo = max(0, np.floor(lo) - 2)
        hi = min(100, np.ceil(hi) + 2)
        for ci in range(3):
            axes[ri][ci].set_ylim(lo, hi)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=13,
               frameon=True, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Test accuracy vs training epoch (5 seeds, mean $\\pm$ 1 std)", fontsize=17, y=1.01)
    # tighter inter-panel padding -> the plot areas fill more of the same footprint
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.2, h_pad=0.4)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT_DIR, f"training_curves.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("Saved: %s" % p)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
