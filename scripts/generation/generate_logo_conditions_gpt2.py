"""
generate_logo_conditions_gpt2.py - same A/B logo-fidelity test as
generate_logo_conditions.py, but on gpt-image-2-i2i instead of flux-2, to
decide which model create_garment_dress_task should standardize on.

Per Rule 8 (brain_modules.txt / hero_generator.py header): gpt-2 needs
MINIMAL prompts, no negative_prompt param exists for it -- so these prompts
are intentionally shorter than the flux-2 versions, respecting gpt-2's own
documented best practice rather than forcing an identical verbose prompt
onto a model known to drift under it.

Condition A (gpt2) - prompt-only, hero_front as sole reference.
Condition B (gpt2) - hero_front + real jacket photo as two references.

gpt-2 only, n=1 each (~6cr x 2 = 12cr).
"""

import sys, json
from pathlib import Path
from datetime import datetime, timezone

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

HERO_FRONT    = Path("data/subject_A/hero_ref/gpt-2-test/hero_front.png")
JACKET_REF    = Path("/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png")
IDENTITY_SPEC = Path("data/subject_A/IDENTITY_SPEC.txt")
OUT_DIR       = Path("outputs/mullay_experiment")
LOG_FILE      = Path("logs/PROMPT_LOG.json")

JACKET_DESCRIPTION_SHORT = (
    "Wearing a black technical zip-up jacket. A bold vertical red wordmark "
    "reading \"MULLAY\" runs down the left chest. Half-body shot, entire "
    "jacket visible from waist to head, gray background."
)


def append_log(entry: dict):
    entries = json.load(open(LOG_FILE)) if LOG_FILE.exists() else []
    entries.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def main():
    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(IDENTITY_SPEC.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[upload] hero_front: {HERO_FRONT}")
    hero_url = gen.upload(str(HERO_FRONT))

    # Condition A (gpt2): prompt-only
    out_a = OUT_DIR / "condition_A_gpt2.png"
    if not out_a.exists():
        prompt_a = f"{identity_core}\n{JACKET_DESCRIPTION_SHORT}"
        print("\n[Condition A gpt2] prompt-only...")
        task_id = gen.create_gpt_i2i_task(prompt_a, [hero_url])
        result = gen.poll(task_id)
        gen.download(result, out_a)
        append_log({"experiment": "mullay_logo_condition_A_gpt2", "model": "gpt-2",
                     "prompt": prompt_a, "refs": [str(HERO_FRONT)], "out_path": str(out_a),
                     "task_id": task_id, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                     "credits_est": "6cr"})
    else:
        print(f"[skip] {out_a} exists")

    # Condition B (gpt2): hero_front + jacket photo
    out_b = OUT_DIR / "condition_B_gpt2.png"
    if not out_b.exists():
        print(f"\n[upload] jacket ref: {JACKET_REF}")
        jacket_url = gen.upload(str(JACKET_REF))
        prompt_b = (
            f"{identity_core}\n"
            "Wearing the exact black jacket shown in the second reference "
            "image, including its red wordmark and piping, unchanged. "
            "Half-body shot, entire jacket visible from waist to head, gray background."
        )
        print("\n[Condition B gpt2] image-to-image with jacket reference...")
        task_id = gen.create_gpt_i2i_task(prompt_b, [hero_url, jacket_url])
        result = gen.poll(task_id)
        gen.download(result, out_b)
        append_log({"experiment": "mullay_logo_condition_B_gpt2", "model": "gpt-2",
                     "prompt": prompt_b, "refs": [str(HERO_FRONT), str(JACKET_REF)],
                     "out_path": str(out_b), "task_id": task_id,
                     "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "credits_est": "6cr"})
    else:
        print(f"[skip] {out_b} exists")

    print("\nDone.")


if __name__ == "__main__":
    main()
