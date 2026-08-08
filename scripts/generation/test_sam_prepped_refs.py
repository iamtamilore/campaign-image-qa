import sys
from pathlib import Path

import os
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT)
from src.api.hero_generator import HeroGenerator

HERO_FRONT_A  = Path("data/subject_A/hero_ref/flux-2-test/hero_front.png")
SPEC_DIR      = Path("data/subject_A")
REF_ISOLATED  = Path("outputs/mullay_experiment/sam_refs/ref_isolated_white.png")
REF_ANNOTATED = Path("outputs/mullay_experiment/sam_refs/ref_annotated_greenbox.png")
OUT_F = Path("outputs/mullay_experiment/condition_F_isolated_ref.png")
OUT_G = Path("outputs/mullay_experiment/condition_G_annotated_ref.png")


def main():
    gen = HeroGenerator()
    identity_core = HeroGenerator.load_identity_spec(SPEC_DIR)

    subject_url = gen.upload(str(HERO_FRONT_A))
    print(f"[upload] subject ref done")

    # ---------- Condition F: isolated-on-white garment ref ----------
    if not OUT_F.exists():
        iso_url = gen.upload(str(REF_ISOLATED))
        print("[F] dressing with isolated-white ref (validated recipe, only the ref changes)...")
        task_f = gen.create_garment_dress_task(identity_core, subject_url, iso_url)
        gen.download(gen.poll(task_f), OUT_F)
    else:
        print(f"[skip] {OUT_F} exists")

    # ---------- Condition G: annotated garment ref + markup-aware prompt ----------
    if not OUT_G.exists():
        ann_url = gen.upload(str(REF_ANNOTATED))
        prompt = (
            f"{identity_core}\n"
            "Wearing the exact garment shown in the second reference image. "
            "The green rectangle in that reference is an annotation marking the "
            "brand wordmark: reproduce the text inside it exactly, unchanged. "
            "The green rectangle itself is NOT part of the garment -- do not "
            "draw any green box or green lines in the output. "
            "Half-body fashion shot, camera pulled back, framed from mid-thigh "
            "to just above the head so the ENTIRE garment is visible -- do NOT "
            "crop the garment.\nGray background, one image only."
        )
        negative = ("green rectangle, green box, green outline, annotation marks, "
                    "tight headshot crop, cropped logo text, cropped garment hem")
        payload = {
            "model": "flux-2/pro-image-to-image",
            "input": {
                "prompt":          prompt,
                "input_urls":      [subject_url, ann_url],
                "aspect_ratio":    "3:4",
                "resolution":      "1K",
                "nsfw_checker":    False,
                "negative_prompt": negative,
            }
        }
        print("[G] dressing with green-box annotated ref...")
        task_g = gen._post(payload)
        print(f"[flux-2 annotated] task → {task_g}  (~5cr)")
        gen.download(gen.poll(task_g), OUT_G)
    else:
        print(f"[skip] {OUT_G} exists")

    print("\nDone: F (isolated) + G (annotated) saved.")


if __name__ == "__main__":
    main()
