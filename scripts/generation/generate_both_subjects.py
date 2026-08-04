"""
generate_both_subjects.py - generate hero reference sheets for both CA2 subjects.

subject A seed: data/subject_A/hero_ref/new_hero.jpeg   (Pinterest ref 1)/Subject A
subject B seed: data/subject_B/hero_ref/new_hero2.jpg   (Pinterest ref 2)/Subject B

My workflow is as follows:
  Step 1: Pinterest ref → flux-2 → hero_front.png (clean grey studio, same person) [FIRST]
           also saves hero_front_gpt2.png via gpt-image-2-i2i for comparison
  Step 2: hero_front.png → flux-2 → 5 angles [FIRST]
  Fallback: gpt-image-2-image-to-image if flux-2 fails on any angle

Setup:
  key auto-loads from the project-root .env file (KIE_API_KEY=...); no manual export needed
  python generate_both_subjects.py

Outputs:
  data/subject_A/hero_ref/hero_front.png  + 5 angles
  data/subject_B/hero_ref/hero_front.png  + 5 angles

Re-run safe - skips already-generated files.
"""

import sys
from pathlib import Path
import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator


SUBJECTS = {
    "A": {
        "seed":    Path("data/subject_A/hero_ref/new_hero.jpeg"),
        "out_dir": Path("data/subject_A/hero_ref"),
    },
    "B": {
        "seed":    Path("data/subject_B/hero_ref/new_hero2.jpg"),
        "out_dir": Path("data/subject_B/hero_ref"),
    },
}


def main():
    gen     = HeroGenerator()
    results = {}

    for subject_id, cfg in SUBJECTS.items():
        if not cfg["seed"].exists():
            print(f"[ERROR] Seed not found: {cfg['seed']}")
            print(f"  Add the reference image and re-run.")
            continue

        result = gen.generate_character_sheet_from_seed(
            subject_id=f"subject_{subject_id}",
            seed_path=str(cfg["seed"]),
            out_dir=cfg["out_dir"],
        )
        results[subject_id] = result

    print("\n=== ALL SUBJECTS DONE ===")
    for sid, r in results.items():
        status = "✓ both fronts done" if r["saved"] == 2 else f"PARTIAL ({r['saved']}/2 fronts)"
        print(f"  Subject {sid}: {status}")
        if r["missing"]:
            print(f"    Missing fronts - re-run to retry")

    print("\nNext step: check both hero_front.png files:")
    print("  subject_A/hero_ref/flux-2-test/hero_front.png")
    print("  subject_A/hero_ref/gpt-2-test/hero_front.png")
    print("  subject_B/hero_ref/flux-2-test/hero_front.png")
    print("  subject_B/hero_ref/gpt-2-test/hero_front.png")
    print("\nThen: python generate_angle_comparison.py --subject both --n 5")


if __name__ == "__main__":
    main()
