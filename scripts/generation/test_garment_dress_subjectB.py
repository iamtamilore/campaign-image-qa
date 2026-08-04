"""
test_garment_dress_subjectB.py - sanity check for the new
HeroGenerator.create_garment_dress_task (Rule 9 / M16.10 recipe), and a
second real data point on flux-2's garment/print rendering, on Subject B
instead of Subject A (n=1 -> n=2 across two different people).
"""

import sys
from pathlib import Path

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

HERO_FRONT_B  = Path("data/subject_B/hero_ref/flux-2-test/hero_front.png")
JACKET_REF    = Path("/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png")
IDENTITY_SPEC = Path("data/subject_B/IDENTITY_SPEC.txt")
OUT_PATH      = Path("outputs/mullay_experiment/condition_B_subjectB_flux2.png")


def main():
    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(IDENTITY_SPEC.parent)

    print(f"[upload] hero_front (Subject B): {HERO_FRONT_B}")
    subject_url = gen.upload(str(HERO_FRONT_B))
    print(f"[upload] jacket ref: {JACKET_REF}")
    garment_url = gen.upload(str(JACKET_REF))

    task_id = gen.create_garment_dress_task(identity_core, subject_url, garment_url)
    result = gen.poll(task_id)
    gen.download(result, OUT_PATH)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
