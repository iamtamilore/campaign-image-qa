import sys
from pathlib import Path

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

JACKET_REF = Path("/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png")

SUBJECTS = {
    "A": {
        "hero_front": Path("data/subject_A/hero_ref/flux-2-test/hero_front.png"),
        "spec_dir":   Path("data/subject_A"),
        "out":        Path("outputs/mullay_experiment/condition_B_subjectA_flux2_v2.png"),
    },
    "B": {
        "hero_front": Path("data/subject_B/hero_ref/flux-2-test/hero_front.png"),
        "spec_dir":   Path("data/subject_B"),
        "out":        Path("outputs/mullay_experiment/condition_B_subjectB_flux2_v2.png"),
    },
}


def main():
    gen = HeroGenerator()
    print(f"[upload] jacket ref: {JACKET_REF}")
    garment_url = gen.upload(str(JACKET_REF))

    for sid, cfg in SUBJECTS.items():
        if cfg["out"].exists():
            print(f"[skip] Subject {sid}: {cfg['out']} exists")
            continue
        identity_core = HeroGenerator.load_identity_spec(cfg["spec_dir"])
        print(f"\n[upload] Subject {sid} hero_front: {cfg['hero_front']}")
        subject_url = gen.upload(str(cfg["hero_front"]))

        print(f"[dress] Subject {sid}...")
        task_id = gen.create_garment_dress_task(identity_core, subject_url, garment_url)
        result = gen.poll(task_id)
        gen.download(result, cfg["out"])

    print("\nDone.")


if __name__ == "__main__":
    main()
