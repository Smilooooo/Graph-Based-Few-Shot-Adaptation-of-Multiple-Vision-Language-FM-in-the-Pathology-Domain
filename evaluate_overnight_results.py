"""Aggregate test metrics across overnight experiment runs.

Scans <root>/<run-name>/log.txt for each run, extracts the final test
accuracy and macro-F1, groups by (variant, shots), and prints per-seed
plus mean ± std tables.

Run naming convention: <variant>_<shots>shot_seed<seed>
e.g.  main_8shot_seed11111  /  noGCN_16shot_seed22222

Usage:
    python evaluate_overnight_results.py
    python evaluate_overnight_results.py --root output/overnight
    python evaluate_overnight_results.py --root output/overnight --csv results.csv
"""

import argparse
import collections
import csv
import os
import re
import statistics


RUN_PATTERN = re.compile(r"(?P<variant>[A-Za-z0-9_]+?)_(?P<shots>\d+)shot_seed(?P<seed>\d+)$")
ACC_PATTERN = re.compile(r"\*\s*accuracy:\s*([\d.]+)%")
F1_PATTERN = re.compile(r"\*\s*macro_f1:\s*([\d.]+)%")


def parse_log(log_path: str):
    """Return (acc, f1) from the final test eval block, or (None, None)."""
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    accs = ACC_PATTERN.findall(text)
    f1s = F1_PATTERN.findall(text)
    acc = float(accs[-1]) if accs else None
    f1 = float(f1s[-1]) if f1s else None
    return acc, f1


def best_log(run_dir: str):
    """Return path to the most recent log file that contains accuracy, or None."""
    candidates = [
        f for f in os.listdir(run_dir)
        if f == "log.txt" or f.startswith("log.txt-")
    ]
    # Sort by modification time, newest first
    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(run_dir, f)),
        reverse=True,
    )
    for fname in candidates:
        path = os.path.join(run_dir, fname)
        acc, f1 = parse_log(path)
        if acc is not None:
            return path
    return None


def collect_runs(root: str):
    """Return dict {(variant, shots): [(seed, acc, f1), ...]} sorted by seed."""
    groups = collections.defaultdict(list)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory not found: {root}")
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        log_path = best_log(run_dir)
        if log_path is None:
            print(f"[warn] no metrics found in {run_dir}")
            continue
        m = RUN_PATTERN.match(name)
        if not m:
            print(f"[skip] unrecognized run name: {name}")
            continue
        variant = m.group("variant")
        shots = int(m.group("shots"))
        seed = int(m.group("seed"))
        acc, f1 = parse_log(log_path)
        if acc is None:
            print(f"[warn] no metrics found in {log_path}")
            continue
        groups[(variant, shots)].append((seed, acc, f1))
    for key in groups:
        groups[key].sort()
    return groups


def print_per_seed(groups):
    print(f"\n{'Variant':<10}{'Shots':<7}{'Seed':<8}{'Acc':<8}{'F1':<8}")
    print("-" * 41)
    for key in sorted(groups):
        variant, shots = key
        for seed, acc, f1 in groups[key]:
            f1_str = f"{f1:.1f}" if f1 is not None else "-"
            print(f"{variant:<10}{shots:<7}{seed:<8}{acc:<8.1f}{f1_str:<8}")


def print_aggregated(groups):
    print(f"\n=== AGGREGATED (mean ± std) ===")
    print(f"{'Variant':<10}{'Shots':<7}{'N':<4}{'Acc mean ± std':<20}{'F1 mean ± std':<20}")
    print("-" * 61)
    for key in sorted(groups):
        variant, shots = key
        accs = [a for _, a, _ in groups[key] if a is not None]
        f1s = [f for _, _, f in groups[key] if f is not None]
        n = len(accs)
        am = statistics.mean(accs)
        asd = statistics.stdev(accs) if n > 1 else 0.0
        if f1s:
            fm = statistics.mean(f1s)
            fsd = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
            f1_str = f"{fm:.2f} ± {fsd:.2f}"
        else:
            f1_str = "-"
        print(f"{variant:<10}{shots:<7}{n:<4}{am:.2f} ± {asd:.2f}        {f1_str:<20}")


def write_csv(groups, csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "shots", "seed", "accuracy", "macro_f1"])
        for key in sorted(groups):
            variant, shots = key
            for seed, acc, f1 in groups[key]:
                w.writerow([variant, shots, seed, acc, f1 if f1 is not None else ""])
    print(f"\nWrote per-seed CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="output/overnight",
        help="Directory containing run subfolders (default: output/overnight)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write per-seed results as CSV",
    )
    args = parser.parse_args()

    groups = collect_runs(args.root)
    if not groups:
        raise SystemExit(f"No runs found under {args.root}")

    print_per_seed(groups)
    print_aggregated(groups)

    if args.csv:
        write_csv(groups, args.csv)


if __name__ == "__main__":
    main()
