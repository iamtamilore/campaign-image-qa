import sys, json
from pathlib import Path
from datetime import datetime, timezone

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

HERO_FRONT    = Path("data/subject_A/hero_ref/flux-2-test/hero_front.png")
JACKET_REF    = Path("/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png")
IDENTITY_SPEC = Path("data/subject_A/IDENTITY_SPEC.txt")
OUT_DIR       = Path("outputs/mullay_experiment")
LOG_FILE      = Path("logs/PROMPT_LOG.json")

JACKET_DESCRIPTION = (
    "Wearing a black technical zip-up jacket with a stand-up collar. "
    "A bold vertical red wordmark reading exactly \"MULLAY\" (six letters: "
    "M-U-L-L-A-Y) runs down the left side of the chest from shoulder to hem, "
    "in a thick sans-serif font, reading bottom-to-top. Thin red piping trim "
    "along the raglan shoulder seams. Studio product photography."
)

FRAMING = (
    "Half-body fashion shot, camera pulled back, framed from mid-thigh to just "
    "above the head so the ENTIRE jacket is visible including the full zipper "
    "and the hem at the waist -- do NOT crop the jacket, do NOT do a tight "
    "head-and-shoulders portrait crop."
)
NEG_FRAMING = "tight headshot crop, close-up portrait, cropped torso, cropped logo text, cropped jacket hem"


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

    # ---------- Condition A: prompt-only, no jacket image ----------
    out_a = OUT_DIR / "condition_A_prompt_only_v2.png"
    if not out_a.exists():
        prompt_a = f"{identity_core}\n{JACKET_DESCRIPTION}\n{FRAMING}\nGray background, one image only."
        print("\n[Condition A v2] prompt-only, wider framing...")
        task_id = gen.create_flux_angle_task(prompt_a, hero_url, negative=NEG_FRAMING)
        result  = gen.poll(task_id)
        gen.download(result, out_a)
        append_log({
            "experiment": "mullay_logo_condition_A", "model": "flux-2",
            "prompt": prompt_a, "refs": [str(HERO_FRONT)],
            "out_path": str(out_a), "task_id": task_id,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "credits_est": "5cr",
        })
    else:
        print(f"[skip] {out_a} exists")

    # ---------- Condition B: image-to-image with jacket ref ----------
    out_b = OUT_DIR / "condition_B_image_ref_v2.png"
    if not out_b.exists():
        print(f"\n[upload] jacket ref: {JACKET_REF}")
        jacket_url = gen.upload(str(JACKET_REF))
        prompt_b = (
            f"{identity_core}\n"
            "Wearing the exact black technical jacket shown in the second reference "
            "image, including its red vertical wordmark and red piping, unchanged. "
            f"{FRAMING}\nGray background, one image only."
        )
        print("\n[Condition B v2] image-to-image, wider framing...")
        task_id = gen._post({
            "model": "flux-2/pro-image-to-image",
            "input": {
                "prompt": prompt_b,
                "input_urls": [hero_url, jacket_url],
                "aspect_ratio": "3:4",
                "resolution": "1K",
                "nsfw_checker": False,
                "negative_prompt": NEG_FRAMING,
            }
        })
        result = gen.poll(task_id)
        gen.download(result, out_b)
        append_log({
            "experiment": "mullay_logo_condition_B", "model": "flux-2",
            "prompt": prompt_b, "refs": [str(HERO_FRONT), str(JACKET_REF)],
            "out_path": str(out_b), "task_id": task_id,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "credits_est": "5cr",
        })
    else:
        print(f"[skip] {out_b} exists")

    print("\nDone. Next: build Condition C locally (homography-stamp ground truth onto B), then score all three.")


if __name__ == "__main__":
    main()
