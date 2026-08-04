"""
generate_angle_hybrid_gpt2.py - same hybrid idea (seed + hero_front as dual
refs) but on gpt-image-2-i2i instead of flux-2, since gpt-2 was the clear
winner in every prior angle test (see reports/comparison_metrics.json and
outputs/results/hybrid_vs_direct_vs_baseline.json).

Question: does dual-referencing push the already-best model even higher?

gpt-2 only, n=1 per angle (~6cr x 5 = 30cr).

Usage:
  python scripts/generation/generate_angle_hybrid_gpt2.py
"""

import sys, json
from pathlib import Path
from datetime import datetime, timezone

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

ANGLES     = ["3qleft", "3qright", "profileleft", "profileright", "back"]
LOG_FILE   = Path("logs/PROMPT_LOG.json")

SEED          = Path("data/subject_A/hero_ref/new_hero.jpeg")
HERO_FRONT    = Path("data/subject_A/hero_ref/gpt-2-test/hero_front.png")
OUT_DIR       = Path("data/subject_A/hero_ref/gpt-2-hybrid")
IDENTITY_SPEC = Path("data/subject_A/IDENTITY_SPEC.txt")


def append_log(entry: dict):
    entries = json.load(open(LOG_FILE)) if LOG_FILE.exists() else []
    entries.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def main():
    if not SEED.exists() or not HERO_FRONT.exists():
        print(f"[ERROR] need both {SEED} and {HERO_FRONT}")
        return

    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(IDENTITY_SPEC.parent)

    print(f"[upload] seed: {SEED}")
    seed_url = gen.upload(str(SEED))
    print(f"[upload] hero_front (gpt2): {HERO_FRONT}")
    front_url = gen.upload(str(HERO_FRONT))
    ref_urls = [seed_url, front_url]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = gen.get_angle_jobs(identity_core)

    generated = 0
    for angle in ANGLES:
        fname, prompt = next((f, p) for f, p in jobs if angle in f)
        out_path = OUT_DIR / f"hero_{angle}_1.png"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists")
            generated += 1
            continue

        print(f"\n[gpt-2 hybrid] {angle} -- seed + hero_front together")
        try:
            task_id = gen.create_gpt_i2i_task(prompt, ref_urls)
            result  = gen.poll(task_id)
            gen.download(result, out_path)

            result_json = json.loads(result.get("resultJson") or "{}")
            result_urls = result_json.get("resultUrls") or result_json.get("images") or []
            now = datetime.now(timezone.utc)
            append_log({
                "subject": "A", "angle": angle, "sample": 1, "model": "gpt-2-hybrid",
                "prompt": prompt, "pinterest_ref": SEED.name,
                "hero_front_ref": HERO_FRONT.name,
                "out_file": out_path.name, "out_path": str(out_path),
                "task_id": task_id,
                "result_url": result_urls[0] if result_urls else "",
                "date": now.strftime("%Y-%m-%d"), "timestamp": now.strftime("%H:%M:%S UTC"),
                "credits_est": "6cr",
                "experiment": "hybrid_seed_plus_hero_front_gpt2",
            })
            generated += 1
        except Exception as e:
            print(f"[error] {angle}: {e}")

    print(f"\n{generated}/{len(ANGLES)} angles generated -> {OUT_DIR}")


if __name__ == "__main__":
    main()
