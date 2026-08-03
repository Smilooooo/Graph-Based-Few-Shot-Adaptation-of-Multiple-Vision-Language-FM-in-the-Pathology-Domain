"""
Adapted-embedding t-SNE comparison (Duy's §5 ask) — one figure per dataset.

Reads the .npz files written by extract_adapted_embeddings.py and produces, per dataset, a row of five
t-SNE panels colored by class with a boxed silhouette score:
    BiomedCLIP | PLIP | CONCH | MLP Fusion | Cross-FM Instance Graph
The first three are frozen FM features (= the old figure, but per dataset and TEST-only); the last two
are the fused representations. Answers "does the Cross-FM fusion separate classes better than simple MLP
fusion?" via the silhouette scores.

Silhouette is computed on the 2-D t-SNE projection (as in the old figure) — a relative separability
indicator across same-recipe panels, not an absolute property of the feature space.

Usage:
    python visualizations/plot_adapted_tsne.py                 # concat for Cross-FM (default)
    python visualizations/plot_adapted_tsne.py --reduce mean   # average the 3 refined queries instead

ASCII-only console output.
"""

import argparse
import os

# Silence loky's Windows physical-core probe (a subprocess that fails on cp1252) BEFORE sklearn import.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ_DIR = os.path.join(PROJECT_ROOT, "output", "adapted_tsne")
OUT_DIR = os.path.join(PROJECT_ROOT, "visualizations")

DISPLAY = {"lunghist": "LungHist700", "bracs": "BRACS", "iciar": "ICIAR2018"}

# Short class labels per dataset, in label order (reused from plot_tsne_fm_comparison.py).
CLASS_ABBR = {
    "lunghist": ["Adeno. WD", "Adeno. MD", "Adeno. PD",
                 "Normal", "SCC WD", "SCC MD", "SCC PD"],
    "bracs":    ["Normal", "Pathological benign", "Usual ductal hyperplasia",
                 "Flat epithelial atypia", "Atypical ductal hyperplasia",
                 "Ductal carcinoma in situ", "Invasive carcinoma"],
    "iciar":    ["Benign", "In situ carcinoma", "Invasive carcinoma", "Normal"],
}

# Ordered frozen FMs -> fusion methods by increasing graph involvement. NOTE: for MLPGraphAdapter and
# MLPHeGraphAdapter the query is NOT graph-refined (the graph refines the class prototypes); the panel
# shows their JointAdapter fused embedding. Only Cross-FM refines the query embedding itself.
# Split into the two rows of the figure: frozen backbones on top, fusion methods below.
BACKBONE_PANELS = [
    ("emb_biomedclip", "BiomedCLIP"),
    ("emb_plip",       "PLIP"),
    ("emb_conch",      "CONCH"),
]
FUSION_PANELS = [
    ("emb_mlp",        "MLP Fusion"),
    ("emb_mlpga",      "MLPGraphAdapter"),
    ("emb_mlphegraph", "MLPHeGraphAdapter"),
    ("emb_ctfig",      "Cross-FM Instance Graph"),
]
PANELS = BACKBONE_PANELS + FUSION_PANELS


def run_tsne(X, perplexity, seed):
    n = X.shape[0]
    perp = min(perplexity, max(5, n // 4))
    return TSNE(n_components=2, perplexity=perp, random_state=seed, n_jobs=1,
                init="pca", learning_rate="auto").fit_transform(X.astype(np.float64))


def safe_silhouette(xy, labels):
    try:
        return silhouette_score(xy, labels)
    except ValueError:
        return float("nan")


def reduce_ctfig(emb, mode):
    """emb: [N, 3, D] -> [N, 3D] (concat) or [N, D] (mean)."""
    return emb.reshape(emb.shape[0], -1) if mode == "concat" else emb.mean(axis=1)


def plot_dataset(ds_key, args):
    path = os.path.join(NPZ_DIR, f"{ds_key}.npz")
    if not os.path.isfile(path):
        print("  SKIP %s (missing %s)" % (ds_key, path))
        return
    data = np.load(path, allow_pickle=True)
    labels = data["labels"]
    abbr = CLASS_ABBR[ds_key]
    n_classes = len(abbr)
    saved = list(data["classnames"])
    if len(saved) != n_classes:
        print("  WARN %s: %d saved classnames vs %d abbreviations" % (ds_key, len(saved), n_classes))
    print("  %s: %d images, %d classes  (label order: %s)" % (ds_key, len(labels), n_classes, saved))

    backbones = [(k, t) for k, t in BACKBONE_PANELS if k in data.files]
    fusion    = [(k, t) for k, t in FUSION_PANELS if k in data.files]
    cmap = matplotlib.colormaps["tab10"]
    sils = []

    def draw_panel(ax, key, title):
        X = data[key]
        if key == "emb_ctfig":
            X = reduce_ctfig(X, args.reduce)
        xy = run_tsne(X, args.perplexity, args.seed)
        sil = safe_silhouette(xy, labels)
        sils.append((title, sil))
        for c in range(n_classes):
            m = labels == c
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], c=[cmap(c)], s=10, alpha=0.6,
                           edgecolors="none", rasterized=True, label=abbr[c])
        ax.set_title(title, fontsize=12, fontweight="bold")
        sil_str = "%.3f" % sil if not np.isnan(sil) else "N/A"
        ax.text(0.02, 0.98, "Silhouette: %s" % sil_str, transform=ax.transAxes, fontsize=9,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(OUT_DIR, exist_ok=True)

    def make_2x2(group, suffix, suptitle):
        """One 2x2 figure. If the group has <4 panels, the first spare cell holds the
        legend; if it fills all 4, the legend goes in a row below the grid."""
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.6), squeeze=False)
        flat = [ax for r in axes for ax in r]
        for ax, (key, title) in zip(flat, group):
            draw_panel(ax, key, title)
        handles, lbls = flat[0].get_legend_handles_labels()
        if len(group) < len(flat):
            legend_ax = flat[len(group)]
            legend_ax.axis("off")
            legend_ax.legend(handles, lbls, loc="center", fontsize=12, framealpha=0.9,
                             markerscale=1.9, title="Class", title_fontsize=13)
            for ax in flat[len(group) + 1:]:
                ax.axis("off")
            rect = [0, 0, 1, 0.96]
        else:
            # adaptive columns: short labels stay in one row, long labels wrap so the
            # legend never exceeds the figure width (which would shrink the panels).
            maxlen = max((len(l) for l in lbls), default=1)
            avail = 8.4  # usable legend width (in) inside the ~9 in figure
            leg_ncol = max(1, min(n_classes, int(avail / (0.08 * maxlen + 0.45))))
            leg_rows = int(np.ceil(n_classes / leg_ncol))
            fig.legend(handles, lbls, loc="lower center", ncol=leg_ncol, fontsize=10,
                       framealpha=0.9, markerscale=1.8, bbox_to_anchor=(0.5, -0.005))
            rect = [0, 0.035 + 0.033 * leg_rows, 1, 0.96]
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.99)
        fig.tight_layout(rect=rect)
        for ext in ("pdf", "png"):
            p = os.path.join(OUT_DIR, f"adapted_tsne_{ds_key}_{suffix}.{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print("    saved: %s" % p)
        plt.close(fig)

    name = DISPLAY.get(ds_key, ds_key)
    make_2x2(backbones, "fms", "%s (FMs)" % name)
    make_2x2(fusion, "methods", "%s (Methods)" % name)

    # print the silhouette comparison that answers Duy's question (MLP Fusion vs Cross-FM)
    print("    silhouettes -> " + "  ".join("%s=%s" % (t, "%.3f" % s if not np.isnan(s) else "N/A")
                                            for t, s in sils))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DISPLAY), choices=list(DISPLAY))
    ap.add_argument("--reduce", choices=["concat", "mean"], default="concat",
                    help="how to reduce the 3 Cross-FM refined queries")
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for ds in args.datasets:
        plot_dataset(ds, args)


if __name__ == "__main__":
    main()
