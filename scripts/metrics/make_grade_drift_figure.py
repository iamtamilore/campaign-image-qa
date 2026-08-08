import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

IMG_SIZE = (128, 128)

# same angle for both subjects, so the only difference down a column is the subject
# and the only difference across a row is the distortion
SAMPLES = [
    ("Subject A", ROOT / "data/subject_A/styled/flux-2-test_hero_3qleft_1.png"),
    ("Subject B", ROOT / "data/subject_B/styled/flux-2-test_hero_3qleft_1.png"),
]

MODES = [
    (None,           "Compliant",    "original"),
    ("hue_shift",    "Hue-shift",    "+60 deg"),
    ("overexposure", "Overexposure", "x1.8 brightness"),
    ("desaturation", "Desaturation", "x0.1 colour"),
]


def load(path):
    return np.array(Image.open(path).convert("RGB").resize(IMG_SIZE)) / 255.0


def distort(img_arr, mode):
    """Identical to distort() in run_parametrisation_experiments.py."""
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    if mode == "hue_shift":
        hsv = np.array(img.convert("HSV"), dtype=np.uint8)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 256
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    elif mode == "overexposure":
        img = ImageEnhance.Brightness(img).enhance(1.8)
    elif mode == "desaturation":
        img = ImageEnhance.Color(img).enhance(0.1)
    return np.array(img) / 255.0


def main():
    nrow, ncol = len(SAMPLES), len(MODES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.3 * nrow))

    for r, (label, path) in enumerate(SAMPLES):
        if not path.exists():
            raise FileNotFoundError(path)
        base = load(path)
        for c, (mode, name, detail) in enumerate(MODES):
            ax = axes[r, c]
            ax.imshow(distort(base, mode) if mode else base)
            ax.set_xticks([]); ax.set_yticks([])
            # red frame marks the off-grade probes the autoencoder must reject
            for side in ax.spines.values():
                side.set_edgecolor("firebrick" if mode else "black")
                side.set_linewidth(2.0 if mode else 1.0)
            if r == 0:
                ax.set_title(f"{name}\n({detail})", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)

    # deliberately NO suptitle: the figure caption in the report carries the title,
    # an in-image title would duplicate it and breaks IEEE figure convention
    fig.tight_layout()
    out = ROOT / "outputs/plots/grade_drift_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[grade-drift] figure saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
