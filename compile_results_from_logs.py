"""
compile_results_from_logs.py
----------------------------
Compile HeGraph results (individual backbones + ensemble) and the MLP fusion
baseline directly from the per-seed log/result files. No `results_*.json` or
`ensemble_results_*.json` aggregation files are read — those are written by
the orchestration scripts and have proven unreliable when reruns happen.

Sources of truth used by this script:

  Individual HeGraph runs:
    <results-dir>/<dataset>_<shots>shot/<backbone>_seed<seed>/log.txt
      -> "* accuracy: XX.X%"
         "* macro_f1: XX.X%"
         "* kappa: 0.XXXX"
         "* macro_auc: 0.XXXX"

  HeGraph ensemble runs:
    <ensemble-dir>/<dataset>_<shots>shot/seed_<seed>/ensemble_results.txt
      -> "Ensemble Accuracy (Majority Voting): XX.XX%"
         "Ensemble Accuracy (Probability Avg) : XX.XX%"
         followed by indented "  macro_f1: ...", "  kappa: ...", "  macro_auc: ..."

  MLP fusion baseline:
    <mlp-dir>/mlp_fusion_<dataset>_descriptive<mlp-suffix>/<shots>shot_seed<seed>/log.txt
      -> same Dassl-style "* metric: value" lines as individual runs

Usage (one-liner):

    python compile_results_from_logs.py \
        --results-dir   output/final_results_rerun_HeGraph_paper \
        --ensemble-dir  output/final_results_rerun_HeGraph_ensemble \
        --mlp-dir       output \
        --mlp-suffix    _60ep_conch448 \
        --output-dir    output/compiled_results
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np


# ── constants ──────────────────────────────────────────────────────────────────

METRICS = [
    ("accuracy",  "Acc (%)"),
    ("macro_f1",  "Macro F1 (%)"),
    ("kappa",     "Kappa"),
    ("macro_auc", "Macro AUC"),
]
METRIC_KEYS = [k for k, _ in METRICS]

SHOTS_ORDER    = [4, 8, 16]
BACKBONE_ORDER = ["biomedclip", "plip", "conch"]
DATASET_ORDER  = ["lunghist", "bracs", "iciar2018"]

BACKBONE_DISPLAY = {
    "biomedclip": "BiomedCLIP",
    "plip":       "PLIP",
    "conch":      "CONCH",
}
DATASET_DISPLAY = {
    "lunghist":  "LUNGHIST",
    "bracs":     "BRACS",
    "iciar2018": "ICIAR2018",
}

_FOLDER_RE      = re.compile(r"^(?P<dataset>.+?)(?:_descriptive)?_(?P<shots>\d+)shot$")
_BACKBONE_RE    = re.compile(r"^(?P<backbone>[a-zA-Z0-9]+)_seed(?P<seed>\d+)$")
_ENS_SEED_RE    = re.compile(r"^seed_(?P<seed>\d+)$")
_MLP_SEED_RE    = re.compile(r"^(?P<shots>\d+)shot_seed(?P<seed>\d+)$")

# Matches "* accuracy: 38.8%" / "* kappa: 0.2868" (Dassl test() format)
_DASSL_METRIC_RE = re.compile(r"^\*\s*(accuracy|macro_f1|kappa|macro_auc):\s*([\-\d\.]+)%?\s*$")


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_combo_folder(folder_name):
    m = _FOLDER_RE.match(folder_name)
    if not m:
        return None, None
    return m.group("dataset"), int(m.group("shots"))


def agg(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std())


def fmt_mean_std(mean, std, decimals=2):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ── log parsing ────────────────────────────────────────────────────────────────

def parse_dassl_log(log_path):
    """
    Parse a Dassl `log.txt` and return the metrics from its FINAL test block.

    The file may contain multiple "* accuracy: ..." lines (e.g. one per epoch
    when `TEST.FINAL_MODEL = "best_val"` is used or when test runs at multiple
    points). We take the last occurrence of each metric.
    """
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = _DASSL_METRIC_RE.match(line.strip())
                if m:
                    metrics[m.group(1)] = float(m.group(2))  # last write wins
    except OSError:
        pass
    return metrics


def parse_ensemble_results_txt(path):
    """
    Parse an `ensemble_results.txt` (per-seed) and return:
        {
          'individual': {backbone: {metric: val, ...}, ...},
          'mv':         {metric: val, ...},
          'avg':        {metric: val, ...},
        }
    Only keys that are present in the file are returned.
    """
    out = {"individual": {}, "mv": {}, "avg": {}}
    if not os.path.isfile(path):
        return out

    # Section parser: walk the file linearly, tracking which "block" we're in.
    section = None          # 'individual:<backbone>' | 'mv' | 'avg' | None
    current_backbone = None

    # Patterns
    indiv_header_re = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*:\s*([\d\.]+)%\s*$")
    mv_header_re    = re.compile(r"^Ensemble Accuracy \(Majority Voting\)\s*:\s*([\d\.]+)%\s*$")
    avg_header_re   = re.compile(r"^Ensemble Accuracy \(Probability Avg\)\s*:\s*([\d\.]+)%\s*$")
    sub_metric_re   = re.compile(r"^\s+(macro_f1|weighted_f1|kappa|macro_auc)\s*:\s*([\-\d\.]+)%?\s*$")

    with open(path, encoding="utf-8", errors="ignore") as fh:
        in_individual_block = False
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("Individual Accuracies"):
                in_individual_block = True
                section = None
                current_backbone = None
                continue

            mv = mv_header_re.match(stripped)
            if mv:
                in_individual_block = False
                section = "mv"
                current_backbone = None
                out["mv"]["accuracy"] = float(mv.group(1))
                continue

            av = avg_header_re.match(stripped)
            if av:
                in_individual_block = False
                section = "avg"
                current_backbone = None
                out["avg"]["accuracy"] = float(av.group(1))
                continue

            # Inside the individual block, a non-indented or 2-space indented
            # "<backbone>: XX.XX%" line introduces a new backbone.
            if in_individual_block:
                m = indiv_header_re.match(line)
                if m:
                    current_backbone = m.group(1)
                    out["individual"].setdefault(current_backbone, {})
                    out["individual"][current_backbone]["accuracy"] = float(m.group(2))
                    section = "individual"
                    continue

            # Sub-metric lines (indented "  metric: value")
            sm = sub_metric_re.match(line)
            if sm:
                key, val = sm.group(1), float(sm.group(2))
                if section == "individual" and current_backbone is not None:
                    out["individual"][current_backbone][key] = val
                elif section == "mv":
                    out["mv"][key] = val
                elif section == "avg":
                    out["avg"][key] = val
    return out


# ── loaders ────────────────────────────────────────────────────────────────────

def load_individual_results(results_dir):
    """
    data[dataset][num_shots][backbone][metric] = [v1, ...]
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    if not os.path.isdir(results_dir):
        print(f"[warn] individual results dir not found: {results_dir}")
        return data

    for combo_folder in sorted(os.listdir(results_dir)):
        dataset, shots = parse_combo_folder(combo_folder)
        if dataset is None:
            continue
        combo_dir = os.path.join(results_dir, combo_folder)
        if not os.path.isdir(combo_dir):
            continue

        for run_folder in sorted(os.listdir(combo_dir)):
            mb = _BACKBONE_RE.match(run_folder)
            if not mb:
                continue
            backbone = mb.group("backbone")
            log_path = os.path.join(combo_dir, run_folder, "log.txt")
            metrics = parse_dassl_log(log_path)
            for k in METRIC_KEYS:
                if k in metrics:
                    data[dataset][shots][backbone][k].append(metrics[k])
    return data


def load_ensemble_results(ensemble_dir):
    """
    ens[dataset][num_shots][kind][metric] = [v1, ...]
        kind in {'mv', 'avg'} (and we also cross-check individuals here)
    """
    ens = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    if not os.path.isdir(ensemble_dir):
        print(f"[warn] ensemble results dir not found: {ensemble_dir}")
        return ens

    for combo_folder in sorted(os.listdir(ensemble_dir)):
        dataset, shots = parse_combo_folder(combo_folder)
        if dataset is None:
            continue
        combo_dir = os.path.join(ensemble_dir, combo_folder)
        if not os.path.isdir(combo_dir):
            continue

        for seed_folder in sorted(os.listdir(combo_dir)):
            if not _ENS_SEED_RE.match(seed_folder):
                continue
            txt_path = os.path.join(combo_dir, seed_folder, "ensemble_results.txt")
            parsed = parse_ensemble_results_txt(txt_path)
            for kind in ("mv", "avg"):
                for k, v in parsed.get(kind, {}).items():
                    if k in METRIC_KEYS:
                        ens[dataset][shots][kind][k].append(v)
    return ens


def load_mlp_results(mlp_root, mlp_suffix):
    """
    mlp[dataset][num_shots][metric] = [v1, ...]
    """
    mlp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not os.path.isdir(mlp_root):
        print(f"[warn] mlp dir not found: {mlp_root}")
        return mlp

    pattern = re.compile(r"^mlp_fusion_(?P<dataset>.+?)_descriptive" +
                         re.escape(mlp_suffix) + r"$")

    for folder in sorted(os.listdir(mlp_root)):
        m = pattern.match(folder)
        if not m:
            continue
        dataset = m.group("dataset")
        combo_dir = os.path.join(mlp_root, folder)
        if not os.path.isdir(combo_dir):
            continue
        for seed_folder in sorted(os.listdir(combo_dir)):
            sm = _MLP_SEED_RE.match(seed_folder)
            if not sm:
                continue
            shots = int(sm.group("shots"))
            log_path = os.path.join(combo_dir, seed_folder, "log.txt")
            metrics = parse_dassl_log(log_path)
            for k in METRIC_KEYS:
                if k in metrics:
                    mlp[dataset][shots][k].append(metrics[k])
    return mlp


# ── CSV writing ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "type", "dataset", "num_shots", "backbone",
    "accuracy_mean",  "accuracy_std",  "accuracy_n",
    "macro_f1_mean",  "macro_f1_std",  "macro_f1_n",
    "kappa_mean",     "kappa_std",     "kappa_n",
    "macro_auc_mean", "macro_auc_std", "macro_auc_n",
]


def _metric_row(metric_vals):
    out = {}
    for k in METRIC_KEYS:
        vals = metric_vals.get(k, [])
        if vals:
            mean, std = agg(vals)
            out[f"{k}_mean"] = round(mean, 4)
            out[f"{k}_std"]  = round(std, 4)
            out[f"{k}_n"]    = len(vals)
        else:
            out[f"{k}_mean"] = ""
            out[f"{k}_std"]  = ""
            out[f"{k}_n"]    = 0
    return out


def write_csv(ind_data, ens_data, mlp_data, output_path):
    rows = []
    for dataset in DATASET_ORDER:
        for shots in SHOTS_ORDER:
            for backbone in BACKBONE_ORDER:
                vals = ind_data.get(dataset, {}).get(shots, {}).get(backbone, {})
                if not vals:
                    continue
                row = {"type": "individual", "dataset": dataset,
                       "num_shots": shots, "backbone": backbone}
                row.update(_metric_row(vals))
                rows.append(row)

            mv = ens_data.get(dataset, {}).get(shots, {}).get("mv", {})
            if mv:
                row = {"type": "ensemble_majority_vote", "dataset": dataset,
                       "num_shots": shots, "backbone": "ensemble"}
                row.update(_metric_row(mv))
                rows.append(row)

            av = ens_data.get(dataset, {}).get(shots, {}).get("avg", {})
            if av:
                row = {"type": "ensemble_prob_avg", "dataset": dataset,
                       "num_shots": shots, "backbone": "ensemble"}
                row.update(_metric_row(av))
                rows.append(row)

            mlp = mlp_data.get(dataset, {}).get(shots, {})
            if mlp:
                row = {"type": "mlp_fusion", "dataset": dataset,
                       "num_shots": shots, "backbone": "mlp_fusion"}
                row.update(_metric_row(mlp))
                rows.append(row)

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── text writing ───────────────────────────────────────────────────────────────

def _cell(metric_vals, key, dec=2):
    vals = metric_vals.get(key, [])
    if not vals:
        return "-" * 14
    mean, std = agg(vals)
    return fmt_mean_std(mean, std, dec)


def write_text(ind_data, ens_data, mlp_data, output_path):
    lines = []
    lines.append("=" * 100)
    lines.append("HEGRAPH COMPILED RESULTS SUMMARY (parsed from log files)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)

    header = (f"  {'Method':<24}  {'Acc (%)':>16}  {'Macro F1 (%)':>16}"
              f"  {'Kappa':>16}  {'Macro AUC':>16}  N")
    sep    = (f"  {'-'*22:<24}  {'-'*14:>16}  {'-'*14:>16}"
              f"  {'-'*14:>16}  {'-'*14:>16}  -")

    for dataset in DATASET_ORDER:
        if dataset not in ind_data and dataset not in ens_data and dataset not in mlp_data:
            continue
        lines.append(f"\n{'─'*100}")
        lines.append(f"  DATASET: {DATASET_DISPLAY.get(dataset, dataset.upper())}")
        lines.append(f"{'─'*100}")

        for shots in SHOTS_ORDER:
            lines.append(f"\n  {shots}-Shot")
            lines.append(header)
            lines.append(sep)

            for backbone in BACKBONE_ORDER:
                vals = ind_data.get(dataset, {}).get(shots, {}).get(backbone, {})
                if not vals:
                    continue
                disp = BACKBONE_DISPLAY.get(backbone, backbone)
                n = len(vals.get("accuracy", []))
                lines.append(
                    f"  {disp:<24}  {_cell(vals,'accuracy'):>16}  {_cell(vals,'macro_f1'):>16}"
                    f"  {_cell(vals,'kappa',4):>16}  {_cell(vals,'macro_auc',4):>16}  {n}"
                )

            mv = ens_data.get(dataset, {}).get(shots, {}).get("mv", {})
            if mv:
                n = len(mv.get("accuracy", []))
                lines.append(
                    f"  {'Ensemble (MV)':<24}  {_cell(mv,'accuracy'):>16}  {_cell(mv,'macro_f1'):>16}"
                    f"  {_cell(mv,'kappa',4):>16}  {_cell(mv,'macro_auc',4):>16}  {n}"
                )

            av = ens_data.get(dataset, {}).get(shots, {}).get("avg", {})
            if av:
                n = len(av.get("accuracy", []))
                lines.append(
                    f"  {'Ensemble (Prob Avg)':<24}  {_cell(av,'accuracy'):>16}  {_cell(av,'macro_f1'):>16}"
                    f"  {_cell(av,'kappa',4):>16}  {_cell(av,'macro_auc',4):>16}  {n}"
                )

            mlp = mlp_data.get(dataset, {}).get(shots, {})
            if mlp:
                n = len(mlp.get("accuracy", []))
                lines.append(
                    f"  {'MLP Fusion (baseline)':<24}  {_cell(mlp,'accuracy'):>16}  {_cell(mlp,'macro_f1'):>16}"
                    f"  {_cell(mlp,'kappa',4):>16}  {_cell(mlp,'macro_auc',4):>16}  {n}"
                )

    lines.append(f"\n{'='*100}\n")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compile HeGraph + ensemble + MLP fusion results from logs only.")
    parser.add_argument("--results-dir",  default="output/final_results_rerun_HeGraph_paper")
    parser.add_argument("--ensemble-dir", default="output/final_results_rerun_HeGraph_ensemble")
    parser.add_argument("--mlp-dir",      default="output")
    parser.add_argument("--mlp-suffix",   default="_60ep_conch448")
    parser.add_argument("--no-mlp",       action="store_true")
    parser.add_argument("--output-dir",   default="output/compiled_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading individual HeGraph results from per-run log.txt ...")
    ind_data = load_individual_results(args.results_dir)

    print("Loading HeGraph ensemble results from per-seed ensemble_results.txt ...")
    ens_data = load_ensemble_results(args.ensemble_dir)

    if args.no_mlp:
        mlp_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    else:
        print("Loading MLP fusion baseline results from per-run log.txt ...")
        mlp_data = load_mlp_results(args.mlp_dir, args.mlp_suffix)

    # Coverage report
    print("\nCoverage:")
    keys = sorted(set(list(ind_data) + list(ens_data) + list(mlp_data)))
    for dataset in keys:
        for shots in SHOTS_ORDER:
            ind_n = {b: len(ind_data.get(dataset, {}).get(shots, {}).get(b, {}).get("accuracy", []))
                     for b in BACKBONE_ORDER
                     if ind_data.get(dataset, {}).get(shots, {}).get(b)}
            ens_n = {kind: len(ens_data.get(dataset, {}).get(shots, {}).get(kind, {}).get("accuracy", []))
                     for kind in ("mv", "avg")
                     if ens_data.get(dataset, {}).get(shots, {}).get(kind)}
            mlp_n = len(mlp_data.get(dataset, {}).get(shots, {}).get("accuracy", []))
            if ind_n or ens_n or mlp_n:
                print(f"  {dataset} {shots}-shot: ind={ind_n}  ens={ens_n}  mlp_n={mlp_n}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"all_results_logs_{ts}.csv")
    txt_path = os.path.join(args.output_dir, f"all_results_logs_{ts}.txt")

    write_csv(ind_data, ens_data, mlp_data, csv_path)
    write_text(ind_data, ens_data, mlp_data, txt_path)

    print(f"\nSaved:\n  {csv_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
