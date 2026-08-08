import sys, json
from pathlib import Path
from datetime import datetime, timezone

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

HERO_FRONT    = Path("data/subject_A/hero_ref/flux-2-test/hero_front.png")
JACKET_REF    = Path("/home/newtazer/Desktop/NCI/2ND/mullay/tech_pack/MULLAY JACKET 2 (FRONT) copy.png")
TEXT_CROP     = Path("outputs/mullay_experiment/ground_truth_wordmark_v2.png")
COLOR_SWATCH  = Path("outputs/mullay_experiment/color_swatch.png")
IDENTITY_SPEC = Path("data/subject_A/IDENTITY_SPEC.txt")
OUT_DIR       = Path("outputs/mullay_experiment")
LOG_FILE      = Path("logs/PROMPT_LOG.json")

FRAMING = (
    "Half-body fashion shot, camera pulled back, framed from mid-thigh to just "
    "above the head so the ENTIRE jacket is visible including the full zipper "
    "and the hem at the waist -- do NOT crop the jacket, do NOT do a tight "
    "head-and-shoulders portrait crop."
)
NEG_FRAMING = "tight headshot crop, close-up portrait, cropped torso, cropped logo text, cropped jacket hem"

BASE_PROMPT = (
    "Wearing the exact black technical jacket shown in the reference "
    "image(s), including its red vertical wordmark and red piping, unchanged. "
    f"{FRAMING}\nGray background, one image only."
)


def append_log(entry: dict):
    entries = json.load(open(LOG_FILE)) if LOG_FILE.exists() else []
    entries.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def run_condition(gen, name, out_name, ref_urls, prompt, identity_core):
    out_path = OUT_DIR / out_name
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return
    full_prompt = f"{identity_core}\n{prompt}"
    print(f"\n[{name}] {len(ref_urls)} refs...")
    task_id = gen._post({
        "model": "flux-2/pro-image-to-image",
        "input": {
            "prompt": full_prompt,
            "input_urls": ref_urls,
            "aspect_ratio": "3:4",
            "resolution": "1K",
            "nsfw_checker": False,
            "negative_prompt": NEG_FRAMING,
        }
    })
    result = gen.poll(task_id)
    gen.download(result, out_path)
    append_log({
        "experiment": f"mullay_logo_{name}", "model": "flux-2",
        "prompt": full_prompt, "n_refs": len(ref_urls),
        "out_path": str(out_path), "task_id": task_id,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "credits_est": "5cr",
    })


def main():
    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(IDENTITY_SPEC.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[upload] hero_front: {HERO_FRONT}")
    hero_url = gen.upload(str(HERO_FRONT))
    print(f"[upload] jacket ref: {JACKET_REF}")
    jacket_url = gen.upload(str(JACKET_REF))
    print(f"[upload] text crop: {TEXT_CROP}")
    text_url = gen.upload(str(TEXT_CROP))
    print(f"[upload] color swatch: {COLOR_SWATCH}")
    swatch_url = gen.upload(str(COLOR_SWATCH))

    # D1: hero + jacket + text-crop
    run_condition(gen, "D1_tight_text_crop", "condition_D1_text_crop.png",
                  [hero_url, jacket_url, text_url], BASE_PROMPT, identity_core)

    # D2: hero + jacket + text-crop + color swatch
    run_condition(gen, "D2_plus_color_swatch", "condition_D2_color_swatch.png",
                  [hero_url, jacket_url, text_url, swatch_url], BASE_PROMPT, identity_core)

    # E: hero + jacket only (same as B), prompt with natural-language color descriptor
    prompt_e = (
        "Wearing the exact black technical jacket shown in the reference "
        "image, including its red vertical wordmark and red piping, unchanged. "
        "The wordmark and piping are a deep, dark oxblood red -- not bright "
        f"red, not orange-red, a muted dark crimson. {FRAMING}\nGray background, one image only."
    )
    run_condition(gen, "E_natural_language_color", "condition_E_color_prompt.png",
                  [hero_url, jacket_url], prompt_e, identity_core)

    print("\nDone.")


if __name__ == "__main__":
    main()
