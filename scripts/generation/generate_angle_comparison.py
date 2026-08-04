"""
generate_angle_comparison.py - flux-2 vs gpt-image-2 ablation experiment.

Generates N_SAMPLES images per angle for both models for one or both subjects.
Every API call is logged to PROMPT_LOG.json for IEEE report reproducibility section.

Naming convention:
  hero_3qleft_1.png, hero_3qleft_2.png ... hero_3qleft_5.png

Output dirs:
  data/subject_A/hero_ref/flux-2-test/   ← flux-2 samples
  data/subject_A/hero_ref/gpt-2-test/    ← gpt-image-2-i2i samples
  data/subject_B/hero_ref/flux-2-test/   ← (same)
  data/subject_B/hero_ref/gpt-2-test/    ← (same)

REQUIRES: hero_front.png for the subject (run generate_both_subjects.py first for Subject A).
Re-run safe - skips existing files.

Usage:
  key auto-loads from the project-root .env file (KIE_API_KEY=...); no manual export needed
  python generate_angle_comparison.py --subject A --n 5
  python generate_angle_comparison.py --subject B --n 5
  python generate_angle_comparison.py --subject both --n 5
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

N_DEFAULT  = 5
ANGLES     = ["3qleft", "3qright", "profileleft", "profileright", "back"]
LOG_FILE   = Path("logs/PROMPT_LOG.json")

SUBJECTS = {
    "A": {
        "hero_front":    Path("data/subject_A/hero_ref/flux-2-test/hero_front.png"),
        "seed":          Path("data/subject_A/hero_ref/new_hero.jpeg"),
        "flux_dir":      Path("data/subject_A/hero_ref/flux-2-test"),
        "gpt2_dir":      Path("data/subject_A/hero_ref/gpt-2-test"),
        "identity_spec": Path("data/subject_A/IDENTITY_SPEC.txt"),
    },
    "B": {
        "hero_front":    Path("data/subject_B/hero_ref/flux-2-test/hero_front.png"),
        "seed":          Path("data/subject_B/hero_ref/new_hero2.jpg"),
        "flux_dir":      Path("data/subject_B/hero_ref/flux-2-test"),
        "gpt2_dir":      Path("data/subject_B/hero_ref/gpt-2-test"),
        "identity_spec": Path("data/subject_B/IDENTITY_SPEC.txt"),
    },
}


def load_log() -> list:
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def append_log(entry: dict):
    entries = load_log()
    entries.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def angle_to_prompt(gen: HeroGenerator, angle: str, identity_core: str) -> str:
    """Pull the angle prompt with REF 3 identity embedded."""
    jobs = gen.get_angle_jobs(identity_core)
    for fname, prompt in jobs:
        if angle in fname:
            return prompt
    raise ValueError(f"Unknown angle: {angle}")


def generate_n_for_angle_logged(gen: HeroGenerator, angle: str, prompt: str,
                                 ref_url: str, out_dir: Path, n: int,
                                 model: str, subject_id: str,
                                 seed_name: str, hero_front_name: str):
    """Wrapper around generate_n_for_angle that adds richer prompt-log fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for i in range(1, n + 1):
        out_path = out_dir / f"hero_{angle}_{i}.png"
        if out_path.exists():
            print(f"  [skip] {out_path.name} exists")
            generated += 1
            continue

        print(f"  [{model}] {angle} sample {i}/{n}...")
        try:
            if model == "flux-2":
                task_id = gen.create_flux_angle_task(prompt, ref_url)
            else:
                task_id = gen.create_gpt_i2i_task(prompt, [ref_url])

            result = gen.poll(task_id)
            gen.download(result, out_path)

            result_json = json.loads(result.get("resultJson") or "{}")
            result_urls = result_json.get("resultUrls") or result_json.get("images") or []
            result_url  = result_urls[0] if result_urls else ""
            now         = datetime.utcnow()

            append_log({
                "subject":          subject_id,
                "angle":            angle,
                "sample":           i,
                "model":            model,
                "prompt":           prompt,
                "pinterest_ref":    seed_name,
                "hero_front_ref":   hero_front_name,
                "out_file":         out_path.name,
                "out_path":         str(out_path),
                "task_id":          task_id,
                "result_url":       result_url,
                "date":             now.strftime("%Y-%m-%d"),
                "timestamp":        now.strftime("%H:%M:%S UTC"),
                "credits_est":      "5cr" if model == "flux-2" else "6cr",
            })
            generated += 1
        except Exception as e:
            print(f"  [error] {model} {angle} sample {i}: {e}")

    return generated


def run_subject(gen: HeroGenerator, subject_id: str, cfg: dict, n: int):
    hero_front = cfg["hero_front"]
    if not hero_front.exists():
        print(f"\n[ERROR] Subject {subject_id}: hero_front.png not found at {hero_front}")
        print("  Run generate_both_subjects.py first to generate the locked front face.")
        return

    print(f"\n{'='*60}")
    print(f"  Subject {subject_id} - uploading hero_front for angle generation")
    print(f"{'='*60}")

    # Load REF 3 identity spec (brain_modules Module 4 - pre-processing before generation)
    identity_core = HeroGenerator.load_identity_spec(cfg["identity_spec"].parent)
    print(f"\n  [identity] loaded spec from {cfg['identity_spec'].name}")
    print(f"  [identity] core preview: {identity_core[:120]}...")

    ref_url         = gen.upload(str(hero_front))
    seed_name       = cfg["seed"].name
    hero_front_name = hero_front.name

    total_flux = 0
    total_gpt2 = 0

    for angle in ANGLES:
        prompt = angle_to_prompt(gen, angle, identity_core)
        print(f"\n  Angle: {angle}")

        # flux-2 samples
        print(f"  >> flux-2 ({n} samples)")
        total_flux += generate_n_for_angle_logged(
            gen, angle, prompt, ref_url, cfg["flux_dir"], n,
            "flux-2", subject_id, seed_name, hero_front_name)

        # gpt-image-2-i2i samples
        print(f"  >> gpt-image-2-i2i ({n} samples)")
        total_gpt2 += generate_n_for_angle_logged(
            gen, angle, prompt, ref_url, cfg["gpt2_dir"], n,
            "gpt-image-2", subject_id, seed_name, hero_front_name)

    expected = n * len(ANGLES)
    print(f"\n{'='*60}")
    print(f"  Subject {subject_id} done.")
    print(f"  flux-2:         {total_flux}/{expected} images")
    print(f"  gpt-image-2:    {total_gpt2}/{expected} images")
    print(f"  Prompt log:     {LOG_FILE}  ({len(load_log())} entries total)")
    print(f"{'='*60}")

    # ── DOCX CHECKPOINT ──────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DOCX CHECKPOINT - Subject {subject_id} angles generated               ║
╚══════════════════════════════════════════════════════════════╝

Open your .docx and type the following (in your own words):

  Section: "Model Comparison - Subject {subject_id}"

  1. Open the flux-2-test/ folder. Open the gpt-2-test/ folder.
     Look at the same angle across both (start with hero_3qleft_1.png).
     Write: "Visually, the [flux-2 / gpt-2] samples appear to better
     preserve Subject {subject_id}'s face because [what you observe]."

  2. Note any artefacts you see (face blur, double face, extra limbs,
     background bleeds, etc.) - write one sentence per issue.

  3. Note the credit cost: flux-2 = 5cr/image, gpt-2 = 6cr/image.
     Write: "At 5 images per angle × 5 angles = 25 images per model,
     flux-2 costs 125cr total, gpt-image-2 costs 150cr total."

Run compute_angle_metrics.py next to get SSIM + face similarity numbers.
╔══════════════════════════════════════════════════════════════╗
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=["A","B","both"], default="both")
    parser.add_argument("--n",       type=int, default=N_DEFAULT,
                        help=f"Samples per angle per model (default: {N_DEFAULT})")
    args = parser.parse_args()

    gen      = HeroGenerator()
    subjects = ["A","B"] if args.subject == "both" else [args.subject]

    for sid in subjects:
        run_subject(gen, sid, SUBJECTS[sid], args.n)

    print("\nAll done. Next: python compute_angle_metrics.py")


if __name__ == "__main__":
    main()
