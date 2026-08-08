from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

FROZEN = "#cdd8e6"   # frozen / not trained
LEARN = "#f5c86a"    # trainable
DATA = "#dce8d8"
DEC = "#e6bfbf"
BASE = "#e9e9e9"
INK = "#2f2f2f"


def box(ax, x, y, w, h, text, colour, fs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.03",
                                facecolor=colour, edgecolor=INK, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.45, color=INK)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.95, color=INK))


fig, ax = plt.subplots(figsize=(7.0, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 10.6); ax.axis("off")

# ---- row 1: data preparation --------------------------------------------
yd = 9.35
box(ax, 0.30, yd, 2.55, 0.90, "106 raw images\n2 subjects", DATA)
box(ax, 3.35, yd, 2.55, 0.90, "pair_builder\n600 pairs (300/300)", DATA)
box(ax, 6.40, yd, 2.55, 0.90, "preprocessing\n224x224\nImageNet norm", DATA, 7.4)
arrow(ax, 2.85, yd + 0.45, 3.35, yd + 0.45)
arrow(ax, 5.90, yd + 0.45, 6.40, yd + 0.45)

# ---- distribution bus: one clean line, no diagonals ----------------------
BUS = 0.75
ax.add_line(Line2D([7.67, 7.67], [yd, yd - 0.42], color=INK, lw=0.95))
ax.add_line(Line2D([BUS, 7.67], [yd - 0.42, yd - 0.42], color=INK, lw=0.95))
ax.add_line(Line2D([BUS, BUS], [1.62, yd - 0.42], color=INK, lw=0.95))

# ---- identity branch -----------------------------------------------------
ax.text(1.15, 8.32, "IDENTITY CHECK   supervised, labelled pairs",
        fontsize=8.4, weight="bold", color=INK)
box(ax, 1.15, 6.85, 2.15, 0.95, "frozen ResNet50\narm A\n23.5M, fixed", FROZEN, 7.4)
box(ax, 1.15, 5.55, 2.15, 0.95, "frozen ResNet50\narm B\nshared weights", FROZEN, 7.4)
box(ax, 3.85, 6.18, 2.15, 1.00, "Dense 2048->128\n+ L2 normalise\nTRAINABLE", LEARN, 7.4)
box(ax, 6.50, 6.18, 1.85, 1.00, "L1 distance\n|a - b|\n128 values", FROZEN, 7.4)
box(ax, 8.85, 6.18, 1.85, 1.00, "Dense(1)\nsigmoid\nTRAINABLE", LEARN, 7.4)

arrow(ax, BUS, 7.32, 1.15, 7.32)
arrow(ax, BUS, 6.02, 1.15, 6.02)
arrow(ax, 3.30, 7.20, 3.85, 6.95)
arrow(ax, 3.30, 6.15, 3.85, 6.42)
arrow(ax, 6.00, 6.68, 6.50, 6.68)
arrow(ax, 8.35, 6.68, 8.85, 6.68)

# ---- grade branch --------------------------------------------------------
ax.text(1.15, 4.85, "GRADE CHECK   unsupervised, compliant images only",
        fontsize=8.4, weight="bold", color=INK)
box(ax, 1.15, 3.40, 3.20, 1.05,
    "convolutional autoencoder\n128>64>32>16, latent 64\ntrained on compliant only", LEARN, 7.4)
box(ax, 4.85, 3.40, 2.55, 1.05,
    "reconstruction MSE\nvs 95th percentile\nthreshold 0.00503", FROZEN, 7.4)
arrow(ax, BUS, 3.92, 1.15, 3.92)
arrow(ax, 4.35, 3.92, 4.85, 3.92)

# ---- baselines -----------------------------------------------------------
ax.text(1.15, 2.72, "BASELINES   same sealed 90-pair test set",
        fontsize=8.4, weight="bold", color=INK)
box(ax, 1.15, 1.15, 2.95, 1.00, "pixel difference\nthreshold 0.05", BASE, 7.4)
box(ax, 4.60, 1.15, 2.95, 1.00, "kNN on the same\nembeddings, dist 0.6", BASE, 7.4)
arrow(ax, BUS, 1.65, 1.15, 1.65)
ax.add_line(Line2D([BUS, BUS], [1.62, 1.65], color=INK, lw=0.95))
ax.add_line(Line2D([2.62, 2.62], [1.15, 0.85], color=INK, lw=0.95))
ax.add_line(Line2D([2.62, 6.07], [0.85, 0.85], color=INK, lw=0.95))
arrow(ax, 6.07, 0.85, 6.07, 1.15)

# ---- decision ------------------------------------------------------------
box(ax, 9.30, 3.05, 2.35, 2.60,
    "QA GATE\n\npass only if\nidentity >= 0.5\nAND\nMSE < threshold\n\nverdict as JSON", DEC, 7.8)
arrow(ax, 9.78, 6.18, 9.90, 5.65)
arrow(ax, 7.40, 3.92, 9.30, 4.15)

# ---- legend --------------------------------------------------------------
legend = [(LEARN, "trainable: ~262k params, ~1% of network"),
          (FROZEN, "frozen / fixed, no gradient"),
          (DATA, "data preparation"),
          (DEC, "final decision")]
for i, (colour, label) in enumerate(legend):
    y = 2.05 - i * 0.42
    ax.add_patch(FancyBboxPatch((8.35, y), 0.28, 0.22,
                                boxstyle="round,pad=0.008",
                                facecolor=colour, edgecolor=INK, linewidth=0.8))
    ax.text(8.76, y + 0.11, label, fontsize=7.0, va="center", color=INK)

fig.tight_layout()
out = ROOT / "outputs/plots/architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"[architecture] saved -> {out.relative_to(ROOT)}")
