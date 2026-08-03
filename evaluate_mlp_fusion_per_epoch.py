r"""
Evaluate MLP Fusion 60-epoch runs at every saved checkpoint (10, 20, ..., 60).

Resumable design:
- Each (dataset, shots, seed, epoch) is evaluated once and the result is
  appended to ``per_epoch_results.jsonl``.
- On restart, results already in the JSONL are skipped, so a crashed run
  can be resumed safely.
- After every epoch is finished, a ``summary_epoch{NN}.json`` file is
  written using everything currently in the JSONL.
- At the end, a ``final_summary_{ts}.json`` file is written that
  aggregates across all epochs/datasets/shots.

Usage:
    python evaluate_mlp_fusion_per_epoch.py \
        --root "E:\BachelorThesis\Data\data(3)\data" \
        --runs lunghist:configs/datasets/lunghist_descriptive.yaml:output/mlp_fusion_lunghist_descriptive_60ep_conch448 \
                bracs:configs/datasets/bracs_descriptive.yaml:output/mlp_fusion_bracs_descriptive_60ep_conch448 \
                iciar:configs/datasets/iciar2018_descriptive.yaml:output/mlp_fusion_iciar2018_descriptive_60ep_conch448 \
        --shots 4 8 16 \
        --seeds 11111 22222 33333 44444 55555 \
        --epochs 10 20 30 40 50 60 \
        --output output/mlp_fusion_per_epoch_eval
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ACC_RE    = re.compile(r"\* accuracy:\s*([0-9.]+)%",  re.IGNORECASE)
F1_RE     = re.compile(r"\* macro_f1:\s*([0-9.]+)%",   re.IGNORECASE)
KAPPA_RE  = re.compile(r"\* kappa:\s*([0-9.]+)",        re.IGNORECASE)
AUC_RE    = re.compile(r"\* macro_auc:\s*([0-9.]+)",    re.IGNORECASE)


def parse_metric(log_text: str, regex: re.Pattern):
    matches = regex.findall(log_text)
    return float(matches[-1]) if matches else None


def evaluate_checkpoint(args, dataset_cfg: str, run_dir: Path, epoch: int):
    # Use the same Python interpreter that runs this script (the active conda env).
    # Avoids `conda run` so we can use shell=False, which makes `timeout` reliably
    # kill the child process instead of leaving orphaned python.exe instances.
    cmd = [
        sys.executable, "train.py",
        "--root", args.root,
        "--dataset-config-file", dataset_cfg,
        "--config-file", "configs/trainers/mlp_fusion.yaml",
        "--output-dir", str(run_dir),
        "--eval-only",
        "--model-dir", str(run_dir),
        "--load-epoch", str(epoch),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        acc   = parse_metric(text, ACC_RE)
        f1    = parse_metric(text, F1_RE)
        kappa = parse_metric(text, KAPPA_RE)
        auc   = parse_metric(text, AUC_RE)
        return acc, f1, kappa, auc, None
    except subprocess.TimeoutExpired as e:
        stdout_tail = (e.stdout or "")[-800:] if isinstance(e.stdout, str) else ""
        stderr_tail = (e.stderr or "")[-800:] if isinstance(e.stderr, str) else ""
        msg = (
            f"TIMEOUT after {args.timeout}s\n"
            f"--- stdout tail ---\n{stdout_tail}\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )
        print(f"    !! TIMEOUT after {args.timeout}s")
        if stderr_tail.strip():
            print(f"    last stderr: {stderr_tail.strip().splitlines()[-1][:200]}")
        return None, None, None, None, msg.strip()[-1500:]
    except subprocess.CalledProcessError as e:
        msg = (e.stdout or "") + "\n" + (e.stderr or "")
        last_lines = [ln for ln in msg.strip().splitlines() if ln.strip()][-3:]
        print(f"    !! FAILED (exit {e.returncode})")
        for ln in last_lines:
            print(f"    | {ln[:200]}")
        return None, None, None, None, msg.strip()[-1500:]


def load_done(jsonl_path: Path):
    """Return mapping (dataset, shots, seed, epoch) -> accuracy from the JSONL."""
    done = {}
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("accuracy") is None:
                continue
            key = (rec["dataset"], rec["shots"], rec["seed"], rec["epoch"])
            done[key] = rec["accuracy"]
    return done


def append_record(jsonl_path: Path, record: dict):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_all_records(jsonl_path: Path):
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _agg_block(values):
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "n": len(values),
        "values": values,
    }


METRIC_KEYS = ["accuracy", "f1", "kappa", "macro_auc"]
JSONL_KEYS  = ["accuracy", "f1", "kappa", "macro_auc"]


def write_per_epoch_summary(out_dir: Path, records, epoch: int):
    """Write summary for a single epoch across all datasets/shots.

    For each (dataset, shots) pair, aggregates all metrics over all seeds.
    Also writes an `overall` block aggregating across all datasets/shots/seeds.
    """
    by_ds_shots = {k: defaultdict(list) for k in METRIC_KEYS}
    overall     = {k: [] for k in METRIC_KEYS}

    for r in records:
        if r["epoch"] != epoch:
            continue
        for k in METRIC_KEYS:
            v = r.get(k)
            if v is not None:
                by_ds_shots[k][(r["dataset"], r["shots"])].append(v)
                overall[k].append(v)

    all_keys = set()
    for k in METRIC_KEYS:
        all_keys |= set(by_ds_shots[k].keys())

    summary = {}
    for ds, shots in all_keys:
        block = {}
        for k in METRIC_KEYS:
            vals = by_ds_shots[k].get((ds, shots), [])
            b = _agg_block(vals)
            if b:
                block[k] = b
        summary.setdefault(ds, {})[str(shots)] = block

    summary["overall"] = {
        k: {"mean": float(np.mean(v)) if v else None,
             "std":  float(np.std(v))  if v else None,
             "n":    len(v)}
        for k, v in overall.items()
    }

    path = out_dir / f"summary_epoch{epoch:02d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def write_final_summary(out_dir: Path, all_records, epochs, timestamp):
    """Aggregate everything across epochs and rank best epoch per dataset."""
    aggregated = {}
    for r in all_records:
        ds    = r["dataset"]
        shots = r["shots"]
        epoch = r["epoch"]
        slot  = (aggregated
                 .setdefault(ds, {})
                 .setdefault(shots, {})
                 .setdefault(epoch, {k: [] for k in METRIC_KEYS}))
        for k in METRIC_KEYS:
            v = r.get(k)
            if v is not None:
                slot[k].append(v)

    final = {}
    for ds, shots_map in aggregated.items():
        final[ds] = {}
        for shots, ep_map in shots_map.items():
            final[ds][str(shots)] = {}
            for ep, vals in ep_map.items():
                block = {}
                for k in METRIC_KEYS:
                    if vals[k]:
                        block[k] = {
                            "mean": float(np.mean(vals[k])),
                            "std":  float(np.std(vals[k])),
                            "n":    len(vals[k]),
                        }
                final[ds][str(shots)][str(ep)] = block

    final_path = out_dir / f"final_summary_{timestamp}.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 70)
    print("BEST EPOCH PER DATASET (mean accuracy over shots and seeds)")
    print("=" * 70)
    for ds, shots_map in aggregated.items():
        epoch_means = {k: defaultdict(list) for k in METRIC_KEYS}
        for shots, ep_map in shots_map.items():
            for ep, vals in ep_map.items():
                for k in METRIC_KEYS:
                    if vals[k]:
                        epoch_means[k][ep].append(float(np.mean(vals[k])))
        ranked = sorted(
            epoch_means["accuracy"].items(),
            key=lambda kv: np.mean(kv[1]),
            reverse=True,
        )
        print(f"\n{ds}:")
        for ep, acc_vals in ranked:
            parts = [f"acc={np.mean(acc_vals):.2f}%"]
            for k, label in [("f1", "f1"), ("kappa", "kappa"), ("macro_auc", "auc")]:
                v = epoch_means[k].get(ep)
                if v:
                    mean_v = float(np.mean(v))
                    if k in ("kappa", "macro_auc"):
                        parts.append(f"{label}={mean_v:.4f}")
                    else:
                        parts.append(f"{label}={mean_v:.2f}%")
            print(f"  epoch {ep:>2}: " + "  ".join(parts))

    print("\n" + "=" * 70)
    print("DETAILED RESULTS — acc% / f1% / kappa / auc (mean over seeds)")
    print("=" * 70)
    epochs_sorted = sorted(epochs)
    for ds, shots_map in aggregated.items():
        for shots in sorted(shots_map):
            cells = []
            for epoch in epochs_sorted:
                vals = shots_map[shots].get(epoch)
                if vals and vals["accuracy"]:
                    a  = float(np.mean(vals["accuracy"]))
                    f1 = float(np.mean(vals["f1"]))       if vals["f1"]        else float("nan")
                    k  = float(np.mean(vals["kappa"]))    if vals["kappa"]     else float("nan")
                    au = float(np.mean(vals["macro_auc"])) if vals["macro_auc"] else float("nan")
                    cells.append(f"{a:.1f}/{f1:.1f}/{k:.3f}/{au:.3f}")
                else:
                    cells.append("-")
            print(f"{ds:<8} {shots:>5}shot  ep" + " | ep".join(
                f"{e}: {c}" for e, c in zip(epochs_sorted, cells)))

    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="List of NAME:DATASET_CFG:RUN_BASE_DIR")
    parser.add_argument("--shots", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[11111, 22222, 33333, 44444, 55555])
    parser.add_argument("--epochs", type=int, nargs="+",
                        default=[10, 20, 30, 40, 50, 60])
    parser.add_argument("--output", default="output/mlp_fusion_per_epoch_eval")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-evaluation timeout in seconds (default: 600 = 10 min). "
                             "A hung evaluation is killed, marked failed, and the script continues.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "per_epoch_results.jsonl"

    run_entries = []
    for entry in args.runs:
        parts = entry.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid --runs entry: {entry}")
        name, dataset_cfg, base_dir = parts
        run_entries.append((name, dataset_cfg, Path(base_dir)))

    done = load_done(jsonl_path)
    if done:
        print(f"Resuming: {len(done)} prior results loaded from {jsonl_path}")

    # Outer loop is EPOCH so per-epoch summaries become available as soon as
    # an epoch finishes across all datasets, even if a later epoch crashes.
    for epoch in args.epochs:
        print("\n" + "#" * 70)
        print(f"# EPOCH {epoch}")
        print("#" * 70)

        for name, dataset_cfg, base_dir in run_entries:
            print(f"\n[{name}] base={base_dir}")
            for shots in args.shots:
                for seed in args.seeds:
                    key = (name, shots, seed, epoch)
                    if key in done:
                        continue
                    run_dir = base_dir / f"{shots}shot_seed{seed}"
                    ckpt_path = run_dir / "mlp_classifier" / f"model.pth.tar-{epoch}"
                    if not ckpt_path.exists():
                        print(f"  {shots}-shot seed {seed}: missing checkpoint")
                        continue
                    print(f"  {shots}-shot seed {seed}: evaluating epoch {epoch} ...")
                    acc, f1, kappa, auc, err = evaluate_checkpoint(args, dataset_cfg, run_dir, epoch)
                    record = {
                        "dataset": name,
                        "shots": shots,
                        "seed": seed,
                        "epoch": epoch,
                        "accuracy": acc,
                        "f1": f1,
                        "kappa": kappa,
                        "macro_auc": auc,
                        "error": err,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    append_record(jsonl_path, record)
                    if acc is not None:
                        done[key] = acc
                        parts = [f"accuracy: {acc:.2f}%"]
                        if f1    is not None: parts.append(f"f1: {f1:.2f}%")
                        if kappa is not None: parts.append(f"kappa: {kappa:.4f}")
                        if auc   is not None: parts.append(f"auc: {auc:.4f}")
                        print("    " + "  ".join(parts))
                    else:
                        print(f"    FAILED (see error in JSONL)")

        # Write a per-epoch summary now that this epoch is finished.
        all_records = read_all_records(jsonl_path)
        path = write_per_epoch_summary(out_dir, all_records, epoch)
        print(f"\n  per-epoch summary written: {path}")

    # Final summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records = read_all_records(jsonl_path)
    final_path = write_final_summary(out_dir, all_records, args.epochs, timestamp)
    print(f"\nFinal summary written: {final_path}")


if __name__ == "__main__":
    main()
