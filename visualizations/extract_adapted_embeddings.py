"""
Extract per-test-image latent embeddings for the adapted-embedding t-SNE comparison (Duy's §5 ask).

For each dataset (seed 11111, 16-shot), over the TEST set, it saves five representations per image:
  emb_biomedclip / emb_plip / emb_conch : frozen FM features (L2-normalized)
  emb_mlp                               : MLP Fusion learned fusion (171-dim penultimate)
  emb_ctfig                             : Cross-FM Instance Graph refined query, [N, 3, D]
                                          (3 per-FM refined queries; plotting concats or means them)
plus `labels` and `classnames`. One .npz per dataset -> output/adapted_tsne/<dataset>.npz.

Why one trainer only: CTFIG's `_extract_features` and MLP Fusion's `_extract_features_from_images`
both L2-normalize each FM feature then (for MLP) concatenate, so the SAME `query_feats` feed the frozen
panels and the MLP input. We build the CTFIG trainer (encoders + support/text buffers + test loader +
CTFIG model) and load only the MLP *classifier* weights separately.

This is GPU + DPS-env work (loads the 3 FM encoders, dassl, checkpoints). Run it yourself, e.g.:
    conda run -n DPS python visualizations/extract_adapted_embeddings.py \
        --root "E:\\BachelorThesis\\Data\\data(3)\\data"

ASCII-only console output.
"""

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Register datasets + trainers with Dassl
import datasets.lungHist700_tien_descriptive  # noqa: F401
import datasets.bracs_tien_descriptive        # noqa: F401
import datasets.iciar2018_tien_descriptive    # noqa: F401
import trainers.ctfig_mvc_trainer             # noqa: F401  (CrossFMTransductiveInstanceGraphMVC)
import trainers.model_registry                # noqa: F401

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, load_checkpoint

from train import extend_cfg
from trainers.mlp_fusion_baseline import MLPClassifier
from trainers.mlpga_model import JointAdapter  # shared by MLPGraphAdapter + MLPHeGraphAdapter

DATASET_CONFIGS = {
    "lunghist": "configs/datasets/lunghist_descriptive.yaml",
    "bracs":    "configs/datasets/bracs_descriptive.yaml",
    "iciar":    "configs/datasets/iciar2018_descriptive.yaml",
}
CTFIG_CONFIG = "configs/trainers/ctfig_mvc.yaml"


def build_cfg(dataset_key, root, seed, shots):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(os.path.join(PROJECT_ROOT, DATASET_CONFIGS[dataset_key]))
    cfg.merge_from_file(os.path.join(PROJECT_ROOT, CTFIG_CONFIG))
    cfg.DATASET.ROOT = root
    cfg.DATASET.NUM_SHOTS = shots
    cfg.SEED = seed
    cfg.TRAINER.NAME = "CrossFMTransductiveInstanceGraphMVC"
    # scratch output dir so we don't touch the real run dir
    cfg.OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "adapted_tsne", "_scratch")
    cfg.freeze()
    return cfg


def load_mlp(mlp_run_dir, num_classes, epoch, device):
    ckpt_path = os.path.join(mlp_run_dir, "mlp_classifier", f"model.pth.tar-{epoch}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ck = load_checkpoint(ckpt_path)
    mlp = MLPClassifier(input_dim=1536, num_classes=num_classes, dropout=0.1)
    mlp.load_state_dict(ck["state_dict"])
    mlp.to(device).eval()
    return mlp


def load_joint_adapter(run_dir, model_name, epoch, device, hidden=512, common=512):
    """Load only the JointAdapter sub-weights of an MLP(He)GraphAdapter checkpoint.

    For both methods the query embedding is joint_adapter(concat) — the graph refines the class
    prototypes/cache, NOT the query (see mlpga_model.py docstring). So the per-image embedding is just
    the JointAdapter output; no encoders or graph needed. HIDDEN_DIM/COMMON_DIM default to 512 (configs).
    """
    ckpt_path = os.path.join(run_dir, model_name, f"model.pth.tar-{epoch}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ck = load_checkpoint(ckpt_path)
    ja = JointAdapter(in_dim=1536, hidden_dim=hidden, common_dim=common, dropout=0.1)
    sub = {k[len("joint_adapter."):]: v for k, v in ck["state_dict"].items()
           if k.startswith("joint_adapter.")}
    ja.load_state_dict(sub)
    ja.to(device).eval()
    return ja


@torch.no_grad()
def extract_for_dataset(dataset_key, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag = f"{dataset_key}_{args.shots}shot_seed{args.seed}"
    ctfig_run = os.path.join(PROJECT_ROOT, "output", "ctfig_mvc_experiments", run_tag)
    mlp_run   = os.path.join(PROJECT_ROOT, "output", "mlp_fusion_experiments", run_tag)
    mlpga_run = os.path.join(PROJECT_ROOT, "output", "mlpga_experiments", run_tag)
    mlphe_run = os.path.join(PROJECT_ROOT, "output", "mlphegraph_experiments", run_tag)
    for d in (ctfig_run, mlp_run, mlpga_run, mlphe_run):
        if not os.path.isdir(d):
            raise FileNotFoundError("run dir missing: %s" % d)

    print("=" * 70)
    print("Dataset: %s   (seed %d, %d-shot, epoch %d)" % (dataset_key, args.seed, args.shots, args.epoch))
    print("=" * 70)

    set_random_seed(args.seed)
    cfg = build_cfg(dataset_key, args.root, args.seed, args.shots)
    trainer = build_trainer(cfg)                       # builds encoders, support/text buffers, CTFIG model
    trainer.load_model(ctfig_run, epoch=args.epoch)    # loads ctfig_mvc/model.pth.tar-<epoch>
    trainer.set_model_mode("eval")

    mlp = load_mlp(mlp_run, trainer.num_classes, args.epoch, device)
    # penultimate = everything except the final Linear; layers[:-1] -> [B, 170]
    mlp_penult = torch.nn.Sequential(*list(mlp.layers[:-1]))
    # JointAdapter query embeddings for the two JointAdapter methods (graph refines protos, not query)
    mlpga_ja = load_joint_adapter(mlpga_run, "mlpga", args.epoch, device)
    mlphe_ja = load_joint_adapter(mlphe_run, "mlphegraph_model", args.epoch, device)

    # hook the instance graph to capture the refined node features (V_all_refined, T_refined)
    captured = {}
    handle = trainer.model.instance_graph.register_forward_hook(
        lambda m, i, o: captured.__setitem__("V", o[0].detach())
    )

    fm_names = trainer.fm_names  # ["biomedclip", "plip", "conch"]
    vsupport, tprotos = trainer._get_support_and_proto_lists()

    buf = {k: [] for k in ("biomedclip", "plip", "conch", "mlp", "mlpga", "mlphegraph", "ctfig")}
    labels = []

    for batch in trainer.test_loader:
        imgs = batch["img"].to(device)
        labels.append(batch["label"].numpy())
        B = imgs.size(0)

        feats = trainer._extract_features(imgs)          # list[3] of [B, D], L2-normalized
        for name, f in zip(fm_names, feats):
            buf[name].append(f.cpu().numpy())

        concat = torch.cat(feats, dim=-1)                # [B, 1536]
        buf["mlp"].append(mlp_penult(concat).cpu().numpy())        # [B, 170]  MLP Fusion penultimate
        buf["mlpga"].append(mlpga_ja(concat).cpu().numpy())        # [B, 512]  MLPGraphAdapter JointAdapter
        buf["mlphegraph"].append(mlphe_ja(concat).cpu().numpy())   # [B, 512]  MLPHeGraphAdapter JointAdapter

        captured.clear()
        _ = trainer.model(feats, vsupport, tprotos)      # triggers hook
        V = captured["V"]                                # [K, CN+B, D]
        Q = V[:, -B:, :].permute(1, 0, 2).contiguous()   # [B, K=3, D]
        buf["ctfig"].append(Q.cpu().numpy())

    handle.remove()

    out = {
        "emb_biomedclip": np.concatenate(buf["biomedclip"]),
        "emb_plip":       np.concatenate(buf["plip"]),
        "emb_conch":      np.concatenate(buf["conch"]),
        "emb_mlp":        np.concatenate(buf["mlp"]),
        "emb_mlpga":      np.concatenate(buf["mlpga"]),
        "emb_mlphegraph": np.concatenate(buf["mlphegraph"]),
        "emb_ctfig":      np.concatenate(buf["ctfig"]),      # [N, 3, D]
        "labels":         np.concatenate(labels),
        "classnames":     np.array(trainer.class_names, dtype=object),
    }
    n = len(out["labels"])
    for k in ("emb_biomedclip", "emb_plip", "emb_conch", "emb_mlp",
              "emb_mlpga", "emb_mlphegraph", "emb_ctfig"):
        assert out[k].shape[0] == n, "%s: %d != %d" % (k, out[k].shape[0], n)
    print("  extracted %d test images | ctfig emb shape %s | mlp emb dim %d"
          % (n, out["emb_ctfig"].shape, out["emb_mlp"].shape[1]))

    out_dir = os.path.join(PROJECT_ROOT, "output", "adapted_tsne")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_key}.npz")
    np.savez_compressed(path, **out)
    print("  saved: %s" % path)

    # free encoders before the next dataset
    del trainer, mlp, mlp_penult, mlpga_ja, mlphe_ja
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DATASET.ROOT (data dir)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS),
                    choices=list(DATASET_CONFIGS))
    ap.add_argument("--seed", type=int, default=11111)
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--epoch", type=int, default=30)
    args = ap.parse_args()

    for ds in args.datasets:
        extract_for_dataset(ds, args)
    print("\nDone. Now run:  python visualizations/plot_adapted_tsne.py")


if __name__ == "__main__":
    main()
