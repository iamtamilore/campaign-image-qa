"""
pair_builder.py — builds the 600 siamese pairs (flashcards) from my dataset.

what it eats (per subject):
  data/subject_A/hero_ref/   → 3 reference images: pinterest seed + flux-2 front + gpt-2 front
  data/subject_A/styled/     → 50 shots: 5 angles x 10 each (5 flux-2 tries + 5 gpt-2 tries)
  data/subject_B/            → same layout, different person
  total: 53 per subject, 106 images

pair logic (from my CA1 proposal §V):
  POSITIVE pair (label 1) → same subject: her ref photo + her styled shot
    3 refs x 50 styled = 150 per subject → 300 total
  NEGATIVE pair (label 0) → different subjects: A's ref + B's styled (and B's ref + A's styled)
    150 + 150 → 300 total
  the code asks for 800 (400+400) but only 300 of each exist, so it takes everything → 600 pairs

why negatives are style-matched: if subject A only wore teal grades and subject B only dark
grades, the network could cheat — "teal = A" — without ever reading a face. both subjects
appear in the same grades AND both generators' styles, so the only signal left is identity.

output:
  data/pairs/pairs_manifest.json — the list of 600 flashcards (img_a, img_b, label)
"""

import json, random
from pathlib import Path
from itertools import product


#the 4 colour grades used across the styled shots
GRADE_LABELS = ["teal_orange", "warm_portra", "dark_moody", "triX_street"]


#collect all of one subject's photos, return dict with the 2 lists
def load_subject_images(data_dir: Path, subject: str) -> dict:
    "return dict with 'hero_ref' and 'styled' image lists for one subject"
    #grab every png, jpg and jpeg sitting at the top of hero_ref/ (NOT the subfolders)
    hero_ref = sorted((data_dir / subject / "hero_ref").glob("*.png")) + \
               sorted((data_dir / subject / "hero_ref").glob("*.jpg")) + \
               sorted((data_dir / subject / "hero_ref").glob("*.jpeg"))
    #same for the styled campaign shots
    styled   = sorted((data_dir / subject / "styled").glob("*.png")) + \
               sorted((data_dir / subject / "styled").glob("*.jpg")) + \
               sorted((data_dir / subject / "styled").glob("*.jpeg"))
    #turn Path objects into plain strings so they save cleanly into json
    return {"hero_ref": [str(p) for p in hero_ref],
            "styled":   [str(p) for p in styled]}


#make the SAME-person flashcards
def build_positive_pairs(images_A: dict, images_B: dict) -> list[dict]:
    "positive: same subject, her ref x her styled (any angle/grade). label 1"
    pairs = []
    #do subject A first, then subject B
    for subj_images in [images_A, images_B]:
        #every ref photo...
        for ref_img in subj_images["hero_ref"]:
            #...paired with every styled shot of the SAME person
            for styled_img in subj_images["styled"]:
                pairs.append({
                    "img_a": ref_img,
                    "img_b": styled_img,
                    #1 = same subject
                    "label": 1,
                    "pair_type": "positive",
                })
    return pairs


#make the DIFFERENT-person flashcards
def build_negative_pairs(images_A: dict, images_B: dict) -> list[dict]:
    "negative: different subjects, crossed both ways. style-matched so the network cant cheat. label 0"
    pairs = []
    #A's refs x B's styled shots
    for ref_img in images_A["hero_ref"]:
        for styled_img in images_B["styled"]:
            pairs.append({
                "img_a": ref_img,
                "img_b": styled_img,
                #0 = different subject
                "label": 0,
                "pair_type": "negative",
            })
    #and the other direction: B's refs x A's styled shots
    for ref_img in images_B["hero_ref"]:
        for styled_img in images_A["styled"]:
            pairs.append({
                "img_a": ref_img,
                "img_b": styled_img,
                "label": 0,
                "pair_type": "negative",
            })
    return pairs


#the main job: build everything, balance it, shuffle it, save it
def build_pairs(data_dir: str = "data", target_pairs: int = 800,
                random_seed: int = 42) -> list[dict]:
    "build the balanced pair manifest (300 pos + 300 neg = 600), save pairs_manifest.json, return the list"
    data_path = Path(data_dir)
    #load both subjects' photo lists
    images_A = load_subject_images(data_path, "subject_A")
    images_B = load_subject_images(data_path, "subject_B")

    #make all possible flashcards, both kinds
    positives = build_positive_pairs(images_A, images_B)
    negatives = build_negative_pairs(images_A, images_B)

    #seeded random generator so the same pairs come out every run (reproducible)
    rng = random.Random(random_seed)
    #aim for half positive, half negative
    half = target_pairs // 2

    #sample down to a balanced set — min() means: take 400 if they exist, else take all
    #(we have 300 of each, so it takes all 300 + 300 = 600)
    pos_sample = rng.sample(positives, min(half, len(positives)))
    neg_sample = rng.sample(negatives, min(half, len(negatives)))
    all_pairs  = pos_sample + neg_sample
    #shuffle so the network never sees all positives then all negatives in a row
    rng.shuffle(all_pairs)

    #save the manifest, create the pairs/ folder if it dont exist
    out_path = data_path / "pairs" / "pairs_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    #count each kind and show the summary
    n_pos = sum(1 for p in all_pairs if p["label"] == 1)
    n_neg = sum(1 for p in all_pairs if p["label"] == 0)
    print(f"[pairs] {len(all_pairs)} pairs  ({n_pos} pos / {n_neg} neg) → {out_path}")
    return all_pairs


#run this file directly to rebuild the manifest
if __name__ == "__main__":
    pairs = build_pairs()
    print(f"[pairs] manifest done — {len(pairs)} pairs")
