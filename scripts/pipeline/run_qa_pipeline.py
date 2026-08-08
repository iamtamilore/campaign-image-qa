import argparse, json, sys
from pathlib import Path

# ── run-from-anywhere shim (dir reorg 2026-07-07): anchor CWD + imports to project root ──
import os as _os, sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

# ── fix every source of randomness so each run reproduces the same numbers ──
# (augmentation uses np.random, tf shuffles batches, both must be seeded)
import random
import numpy as np
random.seed(42)
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    pass
# ─────────────────────────────────────────────────────────────────────────────


def phase_data():
    "phase 1+2: build the 600 flashcards from the 106 images (see src/data/pair_builder.py)"
    print("\n=== PHASE 1+2: Data Understanding + Preparation ===")
    from src.data.pair_builder import build_pairs
    #asks for 800 but only 600 exist, takes everything, perfectly balanced 300/300
    pairs = build_pairs(data_dir="data", target_pairs=800)
    print(f"[pipeline] {len(pairs)} pairs built.")
    return pairs


def phase_train():
    "phase 3: send both students to school (siamese ~6 min CPU, autoencoder ~2 min)"
    print("\n=== PHASE 3: Modelling ===")
    from src.training.train_siamese import train as train_siamese
    from src.training.train_autoencoder import train as train_ae

    print("[pipeline] Training Siamese network...")
    siamese_summary = train_siamese()

    print("[pipeline] Training Autoencoder...")
    ae_summary = train_ae()

    return siamese_summary, ae_summary


def phase_eval():
    "phase 4+5: exam day, all 3 models sit the same sealed 90-pair test, then the post-mortem"
    print("\n=== PHASE 4+5: Evaluation + Error Analysis ===")
    import numpy as np
    import tensorflow as tf
    from src.evaluation.evaluate import run_full_evaluation
    from src.evaluation.error_analysis import run_error_analysis
    from src.data.preprocessing import PairDataset, load_and_preprocess
    from src.models.baselines import evaluate_pixel_diff_baseline, KNNEmbeddingClassifier
    from src.models.backbone import get_embedding_model, L2Normalize
    from src.models.siamese import L1Distance
    #my custom layers must be handed to keras by name or the saved model wont load
    _custom = {"L2Normalize": L2Normalize, "L1Distance": L1Distance}

    #deal the stacks, same seed as training, so the sealed test cards are the same 90
    dataset    = PairDataset()
    test_pairs = dataset.test_pairs

    # ── student A: siamese (my proposed method) ─────────────────────────────
    siamese_model = tf.keras.models.load_model("outputs/models/siamese_best.keras",
                                               custom_objects=_custom)
    test_ds       = dataset.as_tf_dataset("test", batch_size=16)
    y_true_list, y_prob_list, y_pred_list = [], [], []
    for (a_batch, b_batch), lbl_batch in test_ds:
        #confidence scores 0-1 for every pair in the batch
        probs = siamese_model.predict([a_batch, b_batch], verbose=0).flatten()
        y_prob_list.extend(probs)
        #the default 0.5 cutoff turns each score into a yes/no answer
        y_pred_list.extend((probs >= 0.5).astype(int))
        y_true_list.extend(lbl_batch.numpy())

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred_siamese = np.array(y_pred_list)

    siamese_preds = {"name": "Siamese (proposed)", "labels": y_true,
                     "predictions": y_pred_siamese, "y_prob": y_prob}

    # ── student B: pixel-diff (the tracing-paper kid) ───────────────────────
    from PIL import Image
    #raw pixels, no ImageNet prep, this baseline compares pictures directly
    def _load_raw(path):
        return np.array(Image.open(path).convert("RGB").resize((224, 224)))

    loaded_test = [{"img_a_array": _load_raw(p["img_a"]),
                    "img_b_array": _load_raw(p["img_b"]),
                    "label":       p["label"]} for p in test_pairs]
    pixel_preds = evaluate_pixel_diff_baseline(loaded_test)
    pixel_preds["name"] = "Pixel Difference (baseline)"
    #no confidence scores → no ROC-AUC (shows as N/A in the report table)
    pixel_preds["y_prob"] = None

    # ── student C: kNN (the ruler) ───────────────────────────────────────────
    emb_model = get_embedding_model()
    train_pairs = dataset.train_pairs

    #turn each pair's first image into a ResNet fingerprint
    def _emb(pair_list):
        arrs = np.array([load_and_preprocess(p["img_a"]) for p in pair_list])
        return emb_model.predict(arrs, verbose=0)

    #'fit' just stores the training fingerprints, kNN never learns (lazy learner)
    knn = KNNEmbeddingClassifier(k=5)
    knn.fit(_emb(train_pairs), np.array([p["label"] for p in train_pairs]))

    emb_test_a = _emb(test_pairs)
    emb_test_b = np.array([load_and_preprocess(p["img_b"]) for p in test_pairs])
    emb_test_b = emb_model.predict(emb_test_b, verbose=0)
    y_pred_knn = knn.predict_batch(emb_test_a, emb_test_b)
    knn_preds  = {"name": "kNN (baseline)", "labels": y_true,
                  "predictions": y_pred_knn, "y_prob": None}

    # ── one marker grades all three, then the post-mortem on the misses ─────
    all_metrics = run_full_evaluation(siamese_preds, pixel_preds, knn_preds)
    run_error_analysis(y_true, y_pred_siamese, test_pairs, model_name="siamese")
    return all_metrics


def phase_qa(test_image: str):
    "phase 6 (deployment): one new image in → siamese identity check + AE grade check → JSON verdict out"
    print(f"\n=== PHASE 6: QA Mode, {test_image} ===")
    import numpy as np, tensorflow as tf
    from src.data.preprocessing import load_and_preprocess
    from src.models.autoencoder import reconstruction_error, AE_IMG_SIZE

    if not Path(test_image).exists():
        print(f"[qa] ERROR: {test_image} not found")
        sys.exit(1)

    #load both trained guards (custom layers handed to keras by name)
    from src.models.backbone import L2Normalize
    from src.models.siamese import L1Distance
    _custom  = {"L2Normalize": L2Normalize, "L1Distance": L1Distance}
    siamese  = tf.keras.models.load_model("outputs/models/siamese_best.keras",
                                          custom_objects=_custom)
    ae       = tf.keras.models.load_model("outputs/models/autoencoder_best.keras")

    #the verdict line learned at training time (95th percentile of normal redraw errors)
    with open("outputs/results/ae_threshold.json") as f:
        ae_data = json.load(f)
    ae_threshold = ae_data["threshold_mse"]

    #the identity anchor: first hero reference of subject A
    hero_refs = list(Path("data/subject_A/hero_ref").glob("*.png")) + \
                list(Path("data/subject_A/hero_ref").glob("*.jpg"))
    if not hero_refs:
        print("[qa] No hero ref found in data/subject_A/hero_ref/")
        sys.exit(1)

    hero_emb  = load_and_preprocess(str(hero_refs[0]))
    test_emb  = load_and_preprocess(test_image)

    #guard 1, siamese: same person as the hero?
    score     = float(siamese.predict(
        [np.expand_dims(hero_emb, 0), np.expand_dims(test_emb, 0)], verbose=0)[0][0])

    #guard 2, autoencoder: is the look/grade right? (0-1 pixels for the sigmoid decoder)
    from PIL import Image
    img_ae = np.array(Image.open(test_image).convert("RGB").resize(AE_IMG_SIZE)) / 255.0
    mse    = float(reconstruction_error(ae, np.expand_dims(img_ae, 0))[0])

    #the verdict: must pass BOTH guards
    result = {
        "test_image":          test_image,
        "siamese_score":       round(score, 4),
        "subject_consistent":  score >= 0.5,
        "grade_mse":           round(mse, 6),
        "grade_compliant":     mse <= ae_threshold,
        "ae_threshold":        round(ae_threshold, 6),
        "qa_pass":             score >= 0.5 and mse <= ae_threshold,
    }

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mullay QA Pipeline, CA2 x24335592")
    parser.add_argument("--phase", choices=["all", "data", "train", "eval", "qa"],
                        default="all")
    parser.add_argument("--test-image", default=None,
                        help="Image path for --phase qa")
    args = parser.parse_args()

    if args.phase == "data":
        phase_data()
    elif args.phase == "train":
        phase_train()
    elif args.phase == "eval":
        phase_eval()
    elif args.phase == "qa":
        if not args.test_image:
            print("ERROR: --test-image required for --phase qa")
            sys.exit(1)
        phase_qa(args.test_image)
    elif args.phase == "all":
        #the full CRISP-DM run: pairs → school → exam
        phase_data()
        phase_train()
        phase_eval()
