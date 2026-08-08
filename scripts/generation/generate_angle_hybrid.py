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
HERO_FRONT    = Path("data/subject_A/hero_ref/flux-2-test/hero_front.png")
OUT_DIR       = Path("data/subject_A/hero_ref/flux-2-hybrid")
IDENTITY_SPEC = Path("data/subject_A/IDENTITY_SPEC.txt")


def append_log(entry: dict):
    entries = json.load(open(LOG_FILE)) if LOG_FILE.exists() else []
    entries.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def create_flux_angle_task_multiref(gen: HeroGenerator, prompt: str, ref_urls: list[str]) -> str:
    "same payload as HeroGenerator.create_flux_angle_task, but multiple input_urls"
    payload = {
        "model": "flux-2/pro-image-to-image",
        "input": {
            "prompt":          prompt,
            "input_urls":      ref_urls,
            "aspect_ratio":    "3:4",
            "resolution":      "1K",
            "nsfw_checker":    False,
            "negative_prompt": "",
        }
    }
    task_id = gen._post(payload)
    print(f"[flux-2 hybrid] task -> {task_id}  (~5cr, {len(ref_urls)} refs)")
    return task_id


def main():
    if not SEED.exists() or not HERO_FRONT.exists():
        print(f"[ERROR] need both {SEED} and {HERO_FRONT}")
        return

    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(IDENTITY_SPEC.parent)

    print(f"[upload] seed: {SEED}")
    seed_url = gen.upload(str(SEED))
    print(f"[upload] hero_front: {HERO_FRONT}")
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

        print(f"\n[flux-2 hybrid] {angle} -- seed + hero_front together")
        try:
            task_id = create_flux_angle_task_multiref(gen, prompt, ref_urls)
            result  = gen.poll(task_id)
            gen.download(result, out_path)

            result_json = json.loads(result.get("resultJson") or "{}")
            result_urls = result_json.get("resultUrls") or result_json.get("images") or []
            now = datetime.now(timezone.utc)
            append_log({
                "subject": "A", "angle": angle, "sample": 1, "model": "flux-2-hybrid",
                "prompt": prompt, "pinterest_ref": SEED.name,
                "hero_front_ref": HERO_FRONT.name,
                "out_file": out_path.name, "out_path": str(out_path),
                "task_id": task_id,
                "result_url": result_urls[0] if result_urls else "",
                "date": now.strftime("%Y-%m-%d"), "timestamp": now.strftime("%H:%M:%S UTC"),
                "credits_est": "5cr",
                "experiment": "hybrid_seed_plus_hero_front",
            })
            generated += 1
        except Exception as e:
            print(f"[error] {angle}: {e}")

    print(f"\n{generated}/{len(ANGLES)} angles generated -> {OUT_DIR}")


if __name__ == "__main__":
    main()
