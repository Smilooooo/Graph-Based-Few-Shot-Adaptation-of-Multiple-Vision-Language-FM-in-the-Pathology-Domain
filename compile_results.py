"""
compile_results.py
------------------
Compile individual backbone results and ensemble majority-vote results into
a single CSV and a readable text summary.

Usage:
    python compile_results.py \
        --results-dir output/final_results_rerun_paper \
        --ensemble-json output/ensemble_results_rerun_paper/ensemble_epoch100_20260510_190606.json \
        --output-dir output/compiled_results

The script discovers all results_*.json files in <results-dir> automatically,
so no backbone or dataset arguments are needed.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def find_results_jsons(results_dir):
    """Return all results_*.json paths found recursively under results_dir."""
    found = []
    for root, _dirs, files in os.walk(results_dir):
        for f in files:
            if re.match(r"results_.*\.json$", f):
                found.append(os.path.join(root, f))
    return found


def parse_folder_name(path):
    """
    Extract (dataset, num_shots) from the parent folder name.
    Supports:
        lunghist_4shot           -> ("lunghist", 4)
        lunghist_descriptive_4shot -> ("lunghist", 4)
    """
    folder = os.path.basename(os.path.dirname(path))
    m = re.match(r"(\w+?)(?:_descriptive)?_(\d+)shot$", folder)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def agg(values):
    """Return (mean, std) for a list of floats."""
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std())


# ── load individual results ─────────────────────────────────────────────────────

def load_individual_results(results_dir):
    """
    Returns a nested dict:
        data[dataset][num_shots][backbone] = {
            'accuracy': [v1, v2, ...],
            'macro_f1': [...],
            'kappa':    [...],
            'macro_auc':[...],
        }
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for json_path in find_results_jsons(results_dir):
        dataset, num_shots = parse_folder_name(json_path)
        if dataset is None:
            continue

        with open(json_path) as fh:
            doc = json.load(fh)

        for entry in doc.get("results", []):
            if not entry.get("success", True):
                continue
            backbone = entry["backbone"]
            for metric in ("accuracy", "macro_f1", "kappa", "macro_auc"):
                val = entry.get(metric)
                if val is not None:
                    data[dataset][num_shots][backbone][metric].append(val)

    return data


# ── load ensemble results ───────────────────────────────────────────────────────

def load_ensemble_results(ensemble_json):
    """
    Returns:
        ens[dataset][num_shots] = {
            'majority_vote': [...],
            'macro_f1':      [...],
            'weighted_f1':   [...],
            'kappa':         [...],
            'macro_auc':     [...],
        }
    """
    ens = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(ensemble_json) as fh:
        doc = json.load(fh)
    for entry in doc.get("results", []):
        dataset   = entry["dataset"]
        num_shots = entry["num_shots"]
        mv = entry.get("majority_vote")
        if mv is not None:
            ens[dataset][num_shots]["majority_vote"].append(mv)
        em = entry.get("ensemble_metrics", {})
        for mk in ("macro_f1", "weighted_f1", "kappa", "macro_auc"):
            v = em.get(mk)
            if v is not None:
                ens[dataset][num_shots][mk].append(v)
    return ens


# ── format helpers ──────────────────────────────────────────────────────────────

METRICS = [
    ("accuracy",  "Acc (%)"),
    ("macro_f1",  "Macro F1 (%)"),
    ("kappa",     "Kappa"),
    ("macro_auc", "Macro AUC"),
]

SHOTS_ORDER = [4, 8, 16]
BACKBONE_ORDER = ["biomedclip_paper", "conch_paper", "plip_paper"]
DATASET_ORDER = ["lunghist", "bracs", "iciar2018"]

BACKBONE_DISPLAY = {
    "biomedclip_paper": "BiomedCLIP",
    "conch_paper":      "CONCH",
    "plip_paper":       "PLIP",
}


def fmt_mean_std(mean, std, decimals=2):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ── write CSV ──────────────────────────────────────────────────────────────────

def write_csv(ind_data, ens_data, output_path):
    rows = []

    # Individual rows
    for dataset in DATASET_ORDER:
        for shots in SHOTS_ORDER:
            for backbone in BACKBONE_ORDER:
                metric_vals = ind_data.get(dataset, {}).get(shots, {}).get(backbone, {})
                if not metric_vals:
                    continue
                row = {
                    "type": "individual",
                    "dataset": dataset,
                    "num_shots": shots,
                    "backbone": backbone,
                }
                for metric_key, _ in METRICS:
                    vals = metric_vals.get(metric_key, [])
                    if vals:
                        m, s = agg(vals)
                        row[f"{metric_key}_mean"] = round(m, 4)
                        row[f"{metric_key}_std"]  = round(s, 4)
                        row[f"{metric_key}_n"]    = len(vals)
                    else:
                        row[f"{metric_key}_mean"] = ""
                        row[f"{metric_key}_std"]  = ""
                        row[f"{metric_key}_n"]    = 0
                rows.append(row)

    # Ensemble rows
    for dataset in DATASET_ORDER:
        for shots in SHOTS_ORDER:
            slot = ens_data.get(dataset, {}).get(shots, {})
            vals = slot.get("majority_vote", [])
            if not vals:
                continue
            m, s = agg(vals)
            row = {
                "type": "ensemble_majority_vote",
                "dataset": dataset,
                "num_shots": shots,
                "backbone": "ensemble",
                "accuracy_mean": round(m, 4),
                "accuracy_std":  round(s, 4),
                "accuracy_n":    len(vals),
            }
            for metric_key, _ in METRICS:
                if metric_key == "accuracy":
                    continue
                mv = slot.get(metric_key, [])
                if mv:
                    mm, ms = agg(mv)
                    row[f"{metric_key}_mean"] = round(mm, 4)
                    row[f"{metric_key}_std"]  = round(ms, 4)
                    row[f"{metric_key}_n"]    = len(mv)
                else:
                    row[f"{metric_key}_mean"] = ""
                    row[f"{metric_key}_std"]  = ""
                    row[f"{metric_key}_n"]    = 0
            rows.append(row)

    fieldnames = [
        "type", "dataset", "num_shots", "backbone",
        "accuracy_mean", "accuracy_std", "accuracy_n",
        "macro_f1_mean", "macro_f1_std", "macro_f1_n",
        "kappa_mean",    "kappa_std",    "kappa_n",
        "macro_auc_mean","macro_auc_std","macro_auc_n",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── write text summary ──────────────────────────────────────────────────────────

def write_text(ind_data, ens_data, output_path):
    lines = []
    lines.append("=" * 90)
    lines.append("COMPILED RESULTS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)

    for dataset in DATASET_ORDER:
        lines.append(f"\n{'─'*90}")
        lines.append(f"  DATASET: {dataset.upper()}")
        lines.append(f"{'─'*90}")

        for shots in SHOTS_ORDER:
            lines.append(f"\n  {shots}-Shot")
            lines.append(f"  {'Backbone':<20}  {'Acc (%)':>16}  {'Macro F1 (%)':>16}  {'Kappa':>16}  {'Macro AUC':>16}  N")
            lines.append(f"  {'-'*18:<20}  {'-'*14:>16}  {'-'*14:>16}  {'-'*14:>16}  {'-'*14:>16}  -")

            for backbone in BACKBONE_ORDER:
                metric_vals = ind_data.get(dataset, {}).get(shots, {}).get(backbone, {})
                if not metric_vals:
                    continue
                disp = BACKBONE_DISPLAY.get(backbone, backbone)
                n = len(metric_vals.get("accuracy", []))

                def cell(key, dec=2):
                    vals = metric_vals.get(key, [])
                    if not vals:
                        return "-" * 14
                    m, s = agg(vals)
                    return fmt_mean_std(m, s, dec)

                lines.append(
                    f"  {disp:<20}  {cell('accuracy'):>16}  {cell('macro_f1'):>16}"
                    f"  {cell('kappa', 4):>16}  {cell('macro_auc', 4):>16}  {n}"
                )

            # Ensemble row
            slot = ens_data.get(dataset, {}).get(shots, {})
            ens_vals = slot.get("majority_vote", [])
            if ens_vals:
                m, s = agg(ens_vals)

                def ens_cell(key, dec=2):
                    v = slot.get(key, [])
                    if not v:
                        return "n/a"
                    mm, ms = agg(v)
                    return fmt_mean_std(mm, ms, dec)

                lines.append(
                    f"  {'Ensemble (MV)':<20}  {fmt_mean_std(m, s):>16}"
                    f"  {ens_cell('macro_f1'):>16}  {ens_cell('kappa', 4):>16}  {ens_cell('macro_auc', 4):>16}  {len(ens_vals)}"
                )

    lines.append(f"\n{'='*90}\n")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compile individual and ensemble results.")
    parser.add_argument("--results-dir",   required=True,
                        help="Directory containing per-experiment results_*.json files.")
    parser.add_argument("--ensemble-json", required=True,
                        help="Path to the ensemble JSON file.")
    parser.add_argument("--output-dir",    default="output/compiled_results",
                        help="Where to write compiled CSV and text files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading individual results...")
    ind_data = load_individual_results(args.results_dir)

    print("Loading ensemble results...")
    ens_data = load_ensemble_results(args.ensemble_json)

    # Report what was found
    for dataset in sorted(ind_data):
        for shots in sorted(ind_data[dataset]):
            backbones = sorted(ind_data[dataset][shots].keys())
            n_seeds = {b: len(ind_data[dataset][shots][b].get("accuracy", []))
                       for b in backbones}
            print(f"  {dataset} {shots}-shot: {backbones}  seeds={n_seeds}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(args.output_dir, f"all_results_{ts}.csv")
    txt_path  = os.path.join(args.output_dir, f"all_results_{ts}.txt")

    write_csv(ind_data, ens_data, csv_path)
    write_text(ind_data, ens_data, txt_path)

    print(f"\nSaved:\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
