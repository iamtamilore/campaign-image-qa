"""
test_garment_dress_subjectB_gpt2.py - completes the 2x2 matrix (Subject A/B
x flux-2/gpt-2) for the garment-dressing comparison. Subject A gpt-2 already
exists (condition_B_gpt2.png); this fills in Subject B gpt-2.
"""

import sys
from pathlib import Path

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

HERO_FRONT_B = Path("data/subject_B/hero_ref/gpt-2-test/hero_front.png")
JACKET_REF   = Path("/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png")
SPEC_DIR     = Path("data/subject_B")
OUT_PATH     = Path("outputs/mullay_experiment/condition_B_subjectB_gpt2.png")


def main():
    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(SPEC_DIR)

    print(f"[upload] hero_front (Subject B, gpt2): {HERO_FRONT_B}")
    hero_url = gen.upload(str(HERO_FRONT_B))
    print(f"[upload] jacket ref: {JACKET_REF}")
    jacket_url = gen.upload(str(JACKET_REF))

    prompt = (
        f"{identity_core}\n"
        "Wearing the exact black jacket shown in the second reference "
        "image, including its red wordmark and piping, unchanged. "
        "Half-body shot, entire jacket visible from waist to head, gray background."
    )
    print("\n[gpt2 dress] Subject B...")
    task_id = gen.create_gpt_i2i_task(prompt, [hero_url, jacket_url])
    result = gen.poll(task_id)
    gen.download(result, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
