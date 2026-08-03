"""Shared utilities for multi-seed experiment runners."""

import json
import os
import re
import subprocess
import sys
from datetime import datetime


_METRIC_PATTERNS = {
    "accuracy":  re.compile(r"\* accuracy:\s*([\d.]+)%"),
    "macro_f1":  re.compile(r"\* macro_f1:\s*([\d.]+)%"),
    "kappa":     re.compile(r"\* kappa:\s*([\d.]+)"),
    "macro_auc": re.compile(r"\* macro_auc:\s*([\d.]+)"),
}

# Trainers that evaluate mid-training print this marker before each block.
_EPOCH_MARKER = re.compile(r"Intermediate test evaluation at epoch (\d+)")


def is_run_complete(out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    for fname in os.listdir(out_dir):
        if not fname.startswith("log.txt"):
            continue
        with open(os.path.join(out_dir, fname), encoding="utf-8", errors="ignore") as f:
            if "Finish training" in f.read():
                return True
    return False


def _find_latest_log(out_dir: str) -> str | None:
    if not os.path.isdir(out_dir):
        return None
    logs = [f for f in os.listdir(out_dir) if f.startswith("log.txt")]
    if not logs:
        return None
    logs.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)), reverse=True)
    # Prefer the newest log that actually recorded a finished run. Dassl rotates
    # log.txt -> log.txt-<timestamp> on every rerun, so an interrupted restart can
    # leave an empty log.txt as the newest file; picking it blindly yields no
    # metrics (the real results live in an older, rotated log). Fall back to the
    # newest log overall if none contain the completion marker.
    for f in logs:
        path = os.path.join(out_dir, f)
        with open(path, encoding="utf-8", errors="ignore") as fh:
            if "Finish training" in fh.read():
                return path
    return os.path.join(out_dir, logs[0])


def parse_log_metrics(out_dir: str) -> dict:
    metrics = {k: None for k in _METRIC_PATTERNS}
    log = _find_latest_log(out_dir)
    if log is None:
        return metrics
    with open(log, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for key, pat in _METRIC_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def parse_log_metrics_per_epoch(out_dir: str) -> list:
    """Return one metrics dict per intermediate (in-training) test evaluation.

    Trainers with an ``after_epoch`` eval hook print
    ``=> Intermediate test evaluation at epoch N`` before each block; we parse
    the metrics that follow each marker (up to the next marker) so the per-epoch
    accuracy curve can be recorded alongside the final result.  Returns an empty
    list for methods without intermediate evals (GraphAdapter, HeGraph).
    """
    log = _find_latest_log(out_dir)
    if log is None:
        return []
    with open(log, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    markers = list(_EPOCH_MARKER.finditer(text))
    per_epoch = []
    for i, m in enumerate(markers):
        epoch = int(m.group(1))
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        block = text[start:end]
        metrics = {}
        for key, pat in _METRIC_PATTERNS.items():
            found = pat.findall(block)
            metrics[key] = float(found[0]) if found else None
        per_epoch.append({"epoch": epoch, **metrics})
    return per_epoch


def append_result(jsonl_path: str, record: dict) -> None:
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_completed(jsonl_path: str) -> set:
    """Return set of (dataset, shots, seed[, backbone]) tuples already recorded."""
    done = set()
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("dataset", ""), r.get("shots"), r.get("seed"))
            if "backbone" in r:
                key = key + (r["backbone"],)
            done.add(key)
    return done


def run_and_log(cmd, out_dir, jsonl_path, record_base, env=None):
    """Run a training command, parse results, append to JSONL.

    Returns True if the run succeeded.
    """
    # Force UTF-8 in the child so Windows' cp1252 default cannot break either the
    # YAML config read (yacs) or the log write (Dassl) on non-ASCII characters.
    env = dict(env if env is not None else os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(cmd, env=env)

    # Per-epoch intermediate evaluations (if the trainer logs them): one record
    # each, tagged with "epoch". Empty for methods without in-training eval.
    for ep in parse_log_metrics_per_epoch(out_dir):
        epoch = ep.pop("epoch")
        append_result(jsonl_path, {**record_base, "epoch": epoch, **ep,
                                   "output_dir": out_dir})

    # Final result: last eval block in the log (the deployed last-epoch model).
    metrics = parse_log_metrics(out_dir)
    record = {**record_base, **metrics, "output_dir": out_dir,
              "exit_code": result.returncode}
    append_result(jsonl_path, record)
    if result.returncode != 0:
        print(f"  WARNING: exited with code {result.returncode}")
    return result.returncode == 0
