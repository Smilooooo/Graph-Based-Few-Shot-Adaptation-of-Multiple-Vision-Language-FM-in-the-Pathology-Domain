# Graph-Based Few-Shot Adaptation of Multiple Vision-Language Foundation Models in the Pathology Domain

Code for the bachelor thesis.
Methods implemented:

| Group | Method | Trainer | Config |
|-------|--------|---------|--------|
| Baseline (single FM) | GraphAdapter (GA) | `GraphCLIP_v2` | `configs/trainers/GraphCLIP_v2/{biomedclip,plip,conch}_paper.yaml` |
| Baseline (single FM) | HeGraphAdapter (HeGA) | `HeGraphCLIPPaper` | `configs/trainers/{biomedclip,plip,conch}.yaml` |
| Baseline (fusion) | Majority Vote | — (post-hoc over the single-FM runs) | — |
| Baseline (fusion) | MLP Fusion | `MLPFusionBaseline` | `configs/trainers/mlp_fusion.yaml` |
| Proposed | MLPGraphAdapter (MLPGA) | `MLPGAFaithful` | `configs/trainers/mlpga.yaml` |
| Proposed | MLPHeGraphAdapter (MLPHeGA) | `MLPHeGraphAdapter` | `configs/trainers/mlphegraph.yaml` |
| Proposed | Cross-FM Instance Graph (CFIG) | `CrossFMTransductiveInstanceGraphMVC` | `configs/trainers/ctfig_mvc.yaml` |

---

## 1. Repository layout

```
train.py                     # main entry point (CFIG, MLPGA, MLPHeGA, MLP Fusion)
train_graphadapter.py        # entry point for the GraphAdapter baseline
train_hegraph.py             # entry point for the HeGraphAdapter baseline
run_*_multiseed.py           # per-method runners: loop over datasets x shots x seeds
evaluate_overnight_results.py# aggregate per-run test metrics -> mean +/- std tables
evaluate_graphadapter_ensemble.py  # majority-vote ensemble over the GA runs
evaluate_hegraph_ensemble.py       # majority-vote ensemble over the HeGA runs
run_utils.py                 # shared runner helpers (logging, per-epoch parsing)
configs/trainers/            # trainer/method configs
configs/datasets/            # dataset configs (lunghist, bracs, iciar2018)
trainers/                    # model + trainer implementations
datasets/                    # dataset loaders (read splits from --root)
clip/                        # vendored CLIP tokenizer/model utilities
splits/                      # fixed train/val/test splits + few-shot caches (see §3.1)
visualizations/              # plotting scripts (t-SNE, training curves, result charts)
```

Training outputs are written under `output/<method>_experiments/`.

---

## 2. Installation

Commands are shown for **PowerShell on Windows** (the environment the results were
produced in). Exact package versions are in [`requirements.txt`](requirements.txt).

```powershell
# 1. create an environment (conda or venv)
conda create -n gfsa python=3.11 -y
conda activate gfsa

# 2. install PyTorch matching your CUDA (reference: 2.5.1 + CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. install Dassl (training framework) from source
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..

# 4. remaining Python dependencies
pip install -r requirements.txt
```

**Foundation-model weights.** BiomedCLIP and PLIP are downloaded automatically via
`transformers` on first use. CONCH weights are gated.

---

## 3. Data setup

The datasets are not in this repository:

- LungHist700
- BRACS
- ICIAR2018 / BACH

Place the datasets under one root directory and pass it via `--root`. All commands below
assume this PowerShell variable (quote the path if it contains spaces or parentheses):

```powershell
$DATA = "E:/path/to/your/data/root"      # passed as --root
```

### 3.1 Dataset Splits

The exact train/val/test partitions and few-shot support caches used in the thesis are in [`splits/`](splits/). Using these caches
reproduces the exact partitions and few-shot support sets, otherwise they are regenerated
from the seed and will differ.

The full-dataset split JSONs are renamed in the repo with a dataset suffix to avoid a
filename clash (ICIAR2018 and BRACS both use `split_breast_cancer.json`). When copying
them into your data root you must rename them back to the names the loaders expect:

```
splits/
  split_lung_hist.json                 # LungHist700 full split
  split_breast_cancer_iciar.json       # ICIAR2018 (BACH) full split
  split_breast_cancer_bracs.json       # BRACS full split
  split_fewshot_ga_lunghist/*.pkl      # LungHist700 few-shot caches
  split_fewshot_ga_bach/*.pkl          # ICIAR2018 few-shot caches
  split_fewshot_ga_bracs/*.pkl         # BRACS few-shot caches
```

Each dataset loader reads its split from the data directory (`--root`), not from the
`splits/` folder. Copy each file to the matching location and rename as shown:

| Dataset | This repo | Copy to (under `--root`) |
|---------|-----------|--------------------------|
| LungHist700 | `split_lung_hist.json` | `<lunghist_dir>/split_lung_hist.json` (name unchanged) |
| ICIAR2018 | `split_breast_cancer_iciar.json` | `<bach_root>/split_breast_cancer.json` (**rename**) |
| BRACS | `split_breast_cancer_bracs.json` | `<bracs_dir>/split_breast_cancer.json` (**rename**) |

Few-shot caches : copy each
`split_fewshot_ga_*/` folder's `.pkl` files into that dataset's `split_fewshot_ga/`
directory under `--root`.

---

## 4. Reproducing the thesis results

All runners default to the thesis grid: datasets `lunghist bracs iciar`, shots `4 8 16`,
seeds `11111 22222 33333 44444 55555`. Runs are resumable since a completed run is skipped
on restart. Restrict the grid with `--datasets`, `--shots`, `--seeds` (and `--backbones`
for the single-FM baselines).

### 4.1 Single-FM baselines (GraphAdapter, HeGraphAdapter)

```powershell
# GraphAdapter: all 3 backbones x 3 datasets x 3 shots x 5 seeds
python run_graphadapter_multiseed.py --root $DATA

# HeGraphAdapter: same grid
python run_hegraph_multiseed.py --root $DATA

# aggregate -> per-(variant, shots) mean +/- std tables (final epoch)
python evaluate_overnight_results.py --root output/graphadapter_experiments
python evaluate_overnight_results.py --root output/hegraph_experiments
```

### 4.2 Majority-vote ensembles

These reuse the single-FM runs from §4.1 (no retraining), load each backbone's
checkpoint, and compute the hard-voting ensemble per `(dataset, shots, seed)`:

```powershell
python evaluate_graphadapter_ensemble.py --experiments-dir output/graphadapter_experiments
python evaluate_hegraph_ensemble.py     --experiments-dir output/hegraph_experiments
```

### 4.3 MLP Fusion and the proposed methods (MLP Fusion, MLPGA, MLPHeGA, CFIG)

Train each method over the full grid:

```powershell
python run_mlp_fusion_multiseed.py --root $DATA   # -> output/mlp_fusion_experiments
python run_mlpga_multiseed.py      --root $DATA   # -> output/mlpga_experiments
python run_mlphegraph_multiseed.py --root $DATA   # -> output/mlphegraph_experiments
python run_ctfig_mvc_multiseed.py  --root $DATA   # -> output/ctfig_mvc_experiments
```

Each run logs an intermediate test evaluation at every checkpoint, and the runner records
those into `output/<method>_experiments/results.jsonl` tagged with `"epoch"`. The reported
numbers are the **epoch-30** records averaged over the five seeds (same mean ± std
convention as `evaluate_overnight_results.py`).

To evaluate the epoch-30 checkpoint of a single run explicitly:

```powershell
python train.py --eval-only `
  --model-dir output/ctfig_mvc_experiments/bracs_16shot_seed11111 `
  --load-epoch 30 `
  --trainer CrossFMTransductiveInstanceGraphMVC `
  --config-file configs/trainers/ctfig_mvc.yaml `
  --dataset-config-file configs/datasets/bracs_descriptive.yaml `
  --root $DATA
```

### 4.4 Cross-FM Instance Graph ablations (Table, 16-shot)

Each ablation disables one mechanism relative to the full CFIG model. Run at 16 shots on
all three datasets and write to a separate output directory per variant so the full-model
runs are not overwritten. Report the epoch-30 numbers (§4.3).

```powershell
# no graph / no message passing
python run_ctfig_mvc_multiseed.py --root $DATA --shots 16 `
  --output-base output/ablation_nograph      --extra-opts TRAINER.CTFIG.NUM_GNN_LAYERS 0
# no cross-FM edges
python run_ctfig_mvc_multiseed.py --root $DATA --shots 16 `
  --output-base output/ablation_nocrossfm    --extra-opts TRAINER.CTFIG.USE_CROSSFM_GNN False
# no cross-modal edges
python run_ctfig_mvc_multiseed.py --root $DATA --shots 16 `
  --output-base output/ablation_nocrossmodal --extra-opts TRAINER.CTFIG.USE_CROSSMODAL_GNN False
# no intra-modal edges
python run_ctfig_mvc_multiseed.py --root $DATA --shots 16 `
  --output-base output/ablation_nointramodal --extra-opts TRAINER.CTFIG.USE_INTRAMODAL_GNN False
# no transductive query (inductive)
python run_ctfig_mvc_multiseed.py --root $DATA --shots 16 `
  --output-base output/ablation_noquery      --extra-opts TRAINER.CTFIG.USE_TRANSDUCTIVE False
# no multi-view nodes (collapse augmentations to class prototypes)
python run_ctfig_mvc_multiseed.py --root $DATA --shots 16 `
  --output-base output/ablation_nomultiview  --extra-opts TRAINER.CTFIG.USE_PROTOTYPE_NODES True
```

| Ablation | Config override |
|----------|-----------------|
| No graph (no message passing) | `TRAINER.CTFIG.NUM_GNN_LAYERS 0` |
| No cross-FM edges | `TRAINER.CTFIG.USE_CROSSFM_GNN False` |
| No cross-modal edges | `TRAINER.CTFIG.USE_CROSSMODAL_GNN False` |
| No intra-modal edges | `TRAINER.CTFIG.USE_INTRAMODAL_GNN False` |
| No transductive query | `TRAINER.CTFIG.USE_TRANSDUCTIVE False` |
| No multi-view nodes | `TRAINER.CTFIG.USE_PROTOTYPE_NODES True` |

---

## 5. Where each thesis result comes from

| Thesis item | Produced by |
|-------------|-------------|
| Baseline tables (GA / HeGA per backbone, LungHist / BRACS / ICIAR) | `run_graphadapter_multiseed.py`, `run_hegraph_multiseed.py` → `evaluate_overnight_results.py` (§4.1) |
| Majority-vote rows | `evaluate_graphadapter_ensemble.py`, `evaluate_hegraph_ensemble.py` (§4.2) |
| MLP Fusion rows | `run_mlp_fusion_multiseed.py`, epoch 30 (§4.3) |
| Main-results tables (MLPGA, MLPHeGA, CFIG) | `run_mlpga_multiseed.py`, `run_mlphegraph_multiseed.py`, `run_ctfig_mvc_multiseed.py`, epoch 30 (§4.3) |
| CFIG ablation table (16-shot) | `run_ctfig_mvc_multiseed.py --shots 16` with the ablation flags (§4.4) |
| Training-curves figure | `visualizations/plot_training_curves.py` |
| Adapted-embedding t-SNE figures (FMs + fusion methods) | `visualizations/extract_adapted_embeddings.py` → `visualizations/plot_adapted_tsne.py` |

---

## 6. Environment

The reported numbers were produced in this environment:

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.11.8 |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| transformers / scikit-learn / numpy | 5.2.0 / 1.5.2 / 1.26.4 |

