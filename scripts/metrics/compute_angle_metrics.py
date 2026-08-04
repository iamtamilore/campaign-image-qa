"""
compute_angle_metrics.py - flux-2 vs gpt-image-2 face-consistency comparison.

For each subject × each angle × each model:
 - Load N samples from flux-2-test/ and gpt-2-test/
 - Compare each sample to the hero_front.png (the canonical locked face)
 - Metrics computed:
      SSIM - structural similarity to hero_front (0-1, higher = more similar)
      EMB_SIM - cosine similarity of ResNet50 face embedding to hero_front (0-1)
 - Saves per-sample + averaged results to:
      reports/comparison_metrics.json   (machine-readable)
      reports/comparison_report.txt     (human-readable, for .docx notes)

Usage:
  python compute_angle_metrics.py [--subject A|B|both]

Dependencies: scikit-image, tensorflow (already in requirements.txt)
"""

import sys, json, argparse, warnings
import numpy as np
from pathlib import Path
from PIL import Image

# ── run-from-anywhere shim (dir reorg 2026-07-07): anchor CWD + imports to project root ──
import os as _os, sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

# ── SSIM ─────────────────────────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[warn] scikit-image not installed - SSIM disabled. Run: pip install scikit-image")

# ── ResNet50 embedding ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    _EMB_MODEL = None

    def get_embedding_model():
        global _EMB_MODEL
        if _EMB_MODEL is None:
            print("[metrics] loading ResNet50 backbone (same as CA2 Siamese)...")
            _EMB_MODEL = ResNet50(include_top=False, pooling="avg",
                                  weights="imagenet")
        return _EMB_MODEL

    def embedding(img_path: Path) -> np.ndarray:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)[np.newaxis]
        arr = preprocess_input(arr)
        return get_embedding_model().predict(arr, verbose=0)[0]

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[warn] tensorflow not installed - embedding similarity disabled.")

# ─────────────────────────────────────────────────────────────────────────────

ANGLES   = ["3qleft", "3qright", "profileleft", "profileright", "back"]
N_MAX    = 5

SUBJECTS_CFG = {
    "A": {
        "hero_front": Path("data/subject_A/hero_ref/flux-2-test/hero_front.png"),
        "seed":       Path("data/subject_A/hero_ref/new_hero.jpeg"),
        "flux_dir":   Path("data/subject_A/hero_ref/flux-2-test"),
        "gpt2_dir":   Path("data/subject_A/hero_ref/gpt-2-test"),
    },
    "B": {
        "hero_front": Path("data/subject_B/hero_ref/flux-2-test/hero_front.png"),
        "seed":       Path("data/subject_B/hero_ref/new_hero2.jpg"),
        "flux_dir":   Path("data/subject_B/hero_ref/flux-2-test"),
        "gpt2_dir":   Path("data/subject_B/hero_ref/gpt-2-test"),
    },
}


def load_gray_array(path: Path, size: tuple = (512, 512)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img.convert("L"), dtype=np.float64) / 255.0


def compute_ssim(ref_path: Path, gen_path: Path) -> float | None:
    if not HAS_SKIMAGE:
        return None
    ref = load_gray_array(ref_path)
    gen = load_gray_array(gen_path)
    return float(ssim(ref, gen, data_range=1.0))


def compute_emb_sim(ref_path: Path, gen_path: Path) -> float | None:
    if not HAS_TF:
        return None
    ref_emb = embedding(ref_path)
    gen_emb = embedding(gen_path)
    return cosine_sim(ref_emb, gen_emb)


def collect_samples(test_dir: Path, angle: str, n: int) -> list[Path]:
    found = []
    for i in range(1, n + 1):
        p = test_dir / f"hero_{angle}_{i}.png"
        if p.exists():
            found.append(p)
    # also accept un-numbered files (old format: hero_3qleft.png)
    if not found:
        p = test_dir / f"hero_{angle}.png"
        if p.exists():
            found.append(p)
    return found


def score_angle(hero_front: Path, seed: Path, test_dir: Path, angle: str) -> dict:
    """
    SSIM measured vs hero_front (same-crop structural baseline).
    EMB_SIM measured vs Pinterest seed (true identity ground truth - model-neutral).
    Using seed for EMB_SIM removes the flux-2 bias from using flux-2 front as sole reference.
    """
    samples = collect_samples(test_dir, angle, N_MAX)
    if not samples:
        return {"samples": 0, "ssim": None, "emb_sim": None,
                "ssim_all": [], "emb_sim_all": []}

    ssim_scores = []
    emb_scores  = []
    emb_ref     = seed if seed.exists() else hero_front  # fallback to front if seed missing

    for s in samples:
        sv = compute_ssim(hero_front, s)
        ev = compute_emb_sim(emb_ref, s)
        if sv is not None: ssim_scores.append(sv)
        if ev is not None: emb_scores.append(ev)

    def avg_std(vals):
        if not vals: return None, None
        return float(np.mean(vals)), float(np.std(vals))

    ssim_avg, ssim_std = avg_std(ssim_scores)
    emb_avg,  emb_std  = avg_std(emb_scores)

    return {
        "samples":      len(samples),
        "ssim":         ssim_avg,
        "ssim_std":     ssim_std,
        "ssim_all":     [round(v,4) for v in ssim_scores],
        "emb_sim":      emb_avg,
        "emb_sim_std":  emb_std,
        "emb_sim_all":  [round(v,4) for v in emb_scores],
    }


def score_hero_fronts(seed: Path, flux_front: Path, gpt2_front: Path) -> dict:
    """
    Step 1: seed → hero front for each model.
    EMB_SIM only - SSIM not meaningful across different photos/crops.
    """
    results = {}
    for model, front in [("flux2", flux_front), ("gpt2", gpt2_front)]:
        if not front.exists():
            results[model] = {"emb_sim": None}
            continue
        ev = compute_emb_sim(seed, front) if HAS_TF else None
        results[model] = {"emb_sim": round(ev, 4) if ev else None}
    return results


def run_subject(subject_id: str, cfg: dict) -> dict:
    hero_front = cfg["hero_front"]
    if not hero_front.exists():
        print(f"[skip] Subject {subject_id}: hero_front.png not found - run generation first.")
        return {}

    print(f"\n{'='*60}  Subject {subject_id}")
    results = {}
    seed = cfg.get("seed", hero_front)

    # ── Step 1: seed → hero front (how well did we capture the real person?) ──
    gpt2_front = cfg["gpt2_dir"] / "hero_front.png"
    front_scores = score_hero_fronts(seed, hero_front, gpt2_front)
    results["_hero_front_capture"] = front_scores
    f2_cap = front_scores["flux2"]["emb_sim"]
    g2_cap = front_scores["gpt2"]["emb_sim"]
    print(f"\n  Step 1 - seed → hero front (identity capture)")
    if f2_cap and g2_cap:
        winner = "flux-2" if f2_cap > g2_cap else "gpt-2"
        print(f"    EMB_SIM   flux={f2_cap:.4f}   gpt2={g2_cap:.4f}   → {winner} captured seed better")
    else:
        print(f"    EMB_SIM   flux={f2_cap}   gpt2={g2_cap}  (TF not available)")

    # ── Steps 2+3: seed → angles (identity) + hero_front → angles (consistency) ──
    for angle in ANGLES:
        print(f"\n  angle: {angle}")
        flux_scores = score_angle(hero_front, seed, cfg["flux_dir"], angle)
        gpt2_scores = score_angle(hero_front, seed, cfg["gpt2_dir"], angle)
        results[angle] = {"flux2": flux_scores, "gpt2": gpt2_scores}

        if flux_scores["ssim"] and gpt2_scores["ssim"]:
            winner_ssim = "flux-2" if flux_scores["ssim"] > gpt2_scores["ssim"] else "gpt-2"
            print(f"    SSIM vs hero_front   flux={flux_scores['ssim']:.4f}±{flux_scores['ssim_std']:.4f}  "
                  f"gpt2={gpt2_scores['ssim']:.4f}±{gpt2_scores['ssim_std']:.4f}  → {winner_ssim} wins")
        if flux_scores["emb_sim"] and gpt2_scores["emb_sim"]:
            winner_emb = "flux-2" if flux_scores["emb_sim"] > gpt2_scores["emb_sim"] else "gpt-2"
            print(f"    EMB_SIM vs seed      flux={flux_scores['emb_sim']:.4f}±{flux_scores['emb_sim_std']:.4f}  "
                  f"gpt2={gpt2_scores['emb_sim']:.4f}±{gpt2_scores['emb_sim_std']:.4f}  → {winner_emb} wins")

    return results


def write_reports(all_results: dict):
    Path("reports").mkdir(exist_ok=True)
    json_path = Path("reports/comparison_metrics.json")
    txt_path  = Path("reports/comparison_report.txt")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {json_path}")

    # human-readable report
    lines = ["FLUX-2 vs GPT-IMAGE-2 COMPARISON REPORT", "="*60, ""]
    for subject_id, s_results in all_results.items():
        lines.append(f"SUBJECT {subject_id}")
        lines.append("-"*40)

        # Step 1: hero front capture
        cap = s_results.get("_hero_front_capture", {})
        f2c = cap.get("flux2", {}).get("emb_sim")
        g2c = cap.get("gpt2", {}).get("emb_sim")
        lines.append(f"\n  Step 1 - Seed → Hero Front (identity capture from Pinterest ref)")
        if f2c and g2c:
            winner = "flux-2" if f2c > g2c else "gpt-2"
            lines.append(f"  EMB_SIM:  flux={f2c:.4f}   gpt2={g2c:.4f}   >> {winner} captured seed better")
        lines.append("")

        lines.append(f"  Steps 2+3 - Angle shots (EMB_SIM vs seed | SSIM vs hero_front)")
        for angle, scores in s_results.items():
            if angle.startswith("_"): continue
            f2 = scores["flux2"]
            g2 = scores["gpt2"]
            lines.append(f"\n  Angle: {angle}  (n={f2['samples']} flux / {g2['samples']} gpt2)")
            if f2["ssim"] is not None:
                winner = "flux-2" if f2["ssim"] > g2["ssim"] else "gpt-2"
                lines.append(f"  SSIM:     flux={f2['ssim']:.4f}±{f2['ssim_std']:.4f}   "
                             f"gpt2={g2['ssim']:.4f}±{g2['ssim_std']:.4f}  >> {winner}")
                lines.append(f"    flux samples:  {f2['ssim_all']}")
                lines.append(f"    gpt2 samples:  {g2['ssim_all']}")
            if f2["emb_sim"] is not None:
                winner = "flux-2" if f2["emb_sim"] > g2["emb_sim"] else "gpt-2"
                lines.append(f"  EMB_SIM:  flux={f2['emb_sim']:.4f}±{f2['emb_sim_std']:.4f}  "
                             f"gpt2={g2['emb_sim']:.4f}±{g2['emb_sim_std']:.4f}  >> {winner}")
        lines.append("")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {txt_path}")

    # ── DOCX CHECKPOINT ──────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DOCX CHECKPOINT - metrics computed, time to write your notes ║
╚══════════════════════════════════════════════════════════════╝

Open reports/comparison_report.txt, then open your .docx.

For EACH angle, write (in your own words):
  "For the [angle] shot, flux-2 achieved SSIM = [X] (mean of 5 samples)
   and identity similarity = [X], while gpt-image-2 achieved SSIM = [X]
   and identity similarity = [X]. The [model] better preserved Subject
   [A/B]'s face identity because [your observation]."

Then write a summary paragraph:
  "Across all 5 angles and both subjects, [model] consistently scored
   higher on [metric]. The average SSIM advantage was [X].
   This suggests [model] is the better choice for multi-angle character
   sheet generation from a Pinterest reference image."

This paragraph becomes Table 2 + Section IV-B in your IEEE report.
╔══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=["A","B","both"], default="both")
    args = parser.parse_args()

    subjects   = ["A","B"] if args.subject == "both" else [args.subject]
    all_results = {}

    for sid in subjects:
        r = run_subject(sid, SUBJECTS_CFG[sid])
        if r:
            all_results[sid] = r

    if all_results:
        write_reports(all_results)
    else:
        print("\nNo results to report - generate images first.")


if __name__ == "__main__":
    main()
