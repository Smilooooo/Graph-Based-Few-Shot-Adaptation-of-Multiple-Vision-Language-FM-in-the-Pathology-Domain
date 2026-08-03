"""
HeGraph (single-backbone baseline) multi-seed runner.

Loops over datasets x backbones x shots x seeds (all datasets by default).

Usage
-----
python run_hegraph_multiseed.py --root "E:\\BachelorThesis\\Data\\data(3)\\data"
python run_hegraph_multiseed.py --datasets bracs --root "E:\\BachelorThesis\\Data\\data(3)\\data"
python run_hegraph_multiseed.py --datasets lunghist bracs --backbones conch --shots 4 --seeds 11111

Results: output/hegraph_experiments/{dataset}_{backbone}_{N}shot_seed{S}/
Evaluate: python evaluate_overnight_results.py --root output/hegraph_experiments
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

BACKBONE_CONFIGS = {
    "biomedclip": "configs/trainers/biomedclip.yaml",
    "plip":       "configs/trainers/plip.yaml",
    "conch":      "configs/trainers/conch.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="HeGraph multi-seed runner")
    parser.add_argument("--datasets", type=str, nargs="+",
                        choices=list(DATASET_CONFIGS),
                        default=list(DATASET_CONFIGS))
    parser.add_argument("--root", type=str,
                        default=r"E:\BachelorThesis\Data\data(3)\data")
    parser.add_argument("--backbones", type=str, nargs="+",
                        default=["biomedclip", "plip", "conch"])
    parser.add_argument("--shots",  type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seeds",  type=int, nargs="+",
                        default=[11111, 22222, 33333, 44444, 55555])
    parser.add_argument("--gpu",    type=str, default="0")
    parser.add_argument("--output-base", type=str,
                        default="output/hegraph_experiments")
    parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    jsonl_path = os.path.join(args.output_base, "results.jsonl")
    os.makedirs(args.output_base, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    total = len(args.datasets) * len(args.backbones) * len(args.shots) * len(args.seeds)
    done = 0

    for dataset in args.datasets:
        dataset_cfg = DATASET_CONFIGS[dataset]
        for backbone in args.backbones:
            config_file = BACKBONE_CONFIGS[backbone]
            for shots in args.shots:
                for seed in args.seeds:
                    out_dir = os.path.join(
                        args.output_base,
                        f"{dataset}_{backbone}_{shots}shot_seed{seed}",
                    )
                    done += 1
                    if is_run_complete(out_dir):
                        print(f"\n[{done}/{total}] {dataset} {backbone} shots={shots} seed={seed}  SKIPPED")
                        continue

                    cmd = [
                        sys.executable, "train_hegraph.py",
                        "--trainer",             "HeGraphCLIPPaper",
                        "--config-file",         config_file,
                        "--dataset-config-file", dataset_cfg,
                        "--root",                args.root,
                        "--output-dir",          out_dir,
                        "DATASET.NUM_SHOTS",     str(shots),
                        "SEED",                  str(seed),
                        *args.extra_opts,
                    ]
                    print(f"\n[{done}/{total}] {dataset} {backbone} shots={shots} seed={seed}  ->  {out_dir}")
                    run_and_log(cmd, out_dir, jsonl_path,
                                {"method": "HeGraph", "backbone": backbone,
                                 "dataset": dataset, "shots": shots, "seed": seed},
                                env=env)

    print(f"\nAll {total} runs finished.  Results: {jsonl_path}")


if __name__ == "__main__":
    main()
