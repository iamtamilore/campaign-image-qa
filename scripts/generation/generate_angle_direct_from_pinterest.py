import sys, json
from pathlib import Path
from datetime import datetime

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

ANGLES   = ["3qleft", "3qright", "profileleft", "profileright", "back"]
LOG_FILE = Path("logs/PROMPT_LOG.json")

SEED          = Path("data/subject_A/hero_ref/new_hero.jpeg")
OUT_DIR       = Path("data/subject_A/hero_ref/flux-2-direct")
IDENTITY_SPEC = Path("data/subject_A/IDENTITY_SPEC.txt")


def append_log(entry: dict):
    entries = json.load(open(LOG_FILE)) if LOG_FILE.exists() else []
    entries.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def main():
    if not SEED.exists():
        print(f"[ERROR] Seed not found: {SEED}")
        return

    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(IDENTITY_SPEC.parent)
    print(f"[identity] core preview: {identity_core[:120]}...")

    print(f"[upload] {SEED} (direct Pinterest reference, no hero_front hop)")
    ref_url = gen.upload(str(SEED))

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

        print(f"\n[flux-2] {angle}, direct from Pinterest seed")
        try:
            task_id = gen.create_flux_angle_task(prompt, ref_url)
            result  = gen.poll(task_id)
            gen.download(result, out_path)

            result_json = json.loads(result.get("resultJson") or "{}")
            result_urls = result_json.get("resultUrls") or result_json.get("images") or []
            now = datetime.utcnow()
            append_log({
                "subject": "A", "angle": angle, "sample": 1, "model": "flux-2-direct",
                "prompt": prompt, "pinterest_ref": SEED.name,
                "hero_front_ref": None,  # <-- the whole point: no intermediate hop
                "out_file": out_path.name, "out_path": str(out_path),
                "task_id": task_id,
                "result_url": result_urls[0] if result_urls else "",
                "date": now.strftime("%Y-%m-%d"), "timestamp": now.strftime("%H:%M:%S UTC"),
                "credits_est": "5cr",
                "experiment": "direct_from_pinterest_vs_two_stage",
            })
            generated += 1
        except Exception as e:
            print(f"[error] {angle}: {e}")

    print(f"\n{generated}/{len(ANGLES)} angles generated -> {OUT_DIR}")
    print("Next: score with compute_emb_sim(seed, generated) vs reports/comparison_metrics.json")


if __name__ == "__main__":
    main()
