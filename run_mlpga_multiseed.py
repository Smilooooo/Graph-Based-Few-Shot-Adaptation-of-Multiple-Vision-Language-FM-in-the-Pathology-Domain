"""
MLPGAFaithful (multi-backbone GraphAdapter) multi-seed runner.

Usage
-----
# All three datasets (default):
python run_mlpga_multiseed.py --root "E:\\BachelorThesis\\Data\\data(3)\\data"

# A subset:
python run_mlpga_multiseed.py --datasets lunghist bracs --root "E:\\BachelorThesis\\Data\\data(3)\\data"

Results: output/mlpga_experiments/{dataset}_{N}shot_seed{S}/
Evaluate: python evaluate_overnight_results.py --root output/mlpga_experiments
"""

import argparse
import os
import sys

from run_utils import is_run_complete, run_and_log

DATASET_CONFIGS = {
    "lunghist": "configs/datasets/lunghist_descriptive.yaml",
    "bracs":    "configs/datasets/bracs_descriptive.yaml",
    "iciar":    "configs/datasets/iciar2018_descriptive.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="MLPGAFaithful multi-seed runner")
    parser.add_argument("--datasets", type=str, nargs="+", choices=list(DATASET_CONFIGS),
                        default=list(DATASET_CONFIGS))
    parser.add_argument("--root", type=str,
                        default=r"E:\BachelorThesis\Data\data(3)\data")
    parser.add_argument("--shots",  type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds",  type=int, nargs="+",
                        default=[11111, 22222, 33333, 44444, 55555])
    parser.add_argument("--gpu",    type=str, default="0")
    parser.add_argument("--config-file", type=str,
                        default="configs/trainers/mlpga.yaml")
    parser.add_argument("--output-base", type=str,
                        default="output/mlpga_experiments")
    parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    jsonl_path = os.path.join(args.output_base, "results.jsonl")
    os.makedirs(args.output_base, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    total = len(args.datasets) * len(args.shots) * len(args.seeds)
    done = 0

    for dataset in args.datasets:
        dataset_cfg = DATASET_CONFIGS[dataset]
        for shots in args.shots:
            for seed in args.seeds:
                out_dir = os.path.join(
                    args.output_base, f"{dataset}_{shots}shot_seed{seed}"
                )
                done += 1
                if is_run_complete(out_dir):
                    print(f"\n[{done}/{total}] {dataset} shots={shots} seed={seed}  SKIPPED")
                    continue

                cmd = [
                    sys.executable, "train.py",
                    "--trainer",             "MLPGAFaithful",
                    "--config-file",         args.config_file,
                    "--dataset-config-file", dataset_cfg,
                    "--root",                args.root,
                    "--output-dir",          out_dir,
                    "DATASET.NUM_SHOTS",     str(shots),
                    "SEED",                  str(seed),
                    *args.extra_opts,
                ]
                print(f"\n[{done}/{total}] {dataset} shots={shots} seed={seed}  ->  {out_dir}")
                run_and_log(cmd, out_dir, jsonl_path,
                            {"method": "MLPGAFaithful", "dataset": dataset,
                             "shots": shots, "seed": seed}, env=env)

    print(f"\nAll {total} runs finished.  Results: {jsonl_path}")
    print(f"Aggregate: python evaluate_overnight_results.py --root {args.output_base}")


if __name__ == "__main__":
    main()
