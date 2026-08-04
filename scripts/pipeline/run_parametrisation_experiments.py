"""
run_parametrisation_experiments.py - Rubric Row 5 "diverse parametrisation".

Four experiments, all free/local, no Kie.ai credits:
  EXP 1 - Siamese decision threshold: tune on VAL (never test), apply to test.
          ROC-AUC=1.0 predicts a threshold with perfect test F1 exists.
  EXP 2 - kNN distance threshold sweep (0.4/0.5/0.6/0.7/0.8).
          NOTE: baseline's `k` is vestigial - predict_pair uses distance rule only.
  EXP 3 - AE anomaly threshold percentile (90/95/99) on compliant vs
          synthetically-distorted (hue shift / overexposure / desaturation) images.
  EXP 4 - Siamese embedding_dim 64 vs 128 vs 256 (retrains 64 + 256).

Outputs: outputs/results/parametrisation_experiments.json
"""

import json, time
import numpy as np
from pathlib import Path

# ── run-from-anywhere shim (dir reorg 2026-07-07): anchor CWD + imports to project root ──
import os as _os, sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

from PIL import Image, ImageEnhance

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.data.preprocessing import PairDataset, load_and_preprocess
from src.models.backbone import get_embedding_model, L2Normalize
from src.models.siamese import L1Distance
from src.models.autoencoder import reconstruction_error, AE_IMG_SIZE

CUSTOM = {"L2Normalize": L2Normalize, "L1Distance": L1Distance}
RESULTS = {}
t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

def prf(y_true, y_pred):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

dataset = PairDataset()

def score_pairs(model, pairs, batch=16):
    """Return (y_true, y_prob) for a list of pair dicts."""
    probs, labels = [], []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i+batch]
        a = np.array([load_and_preprocess(p["img_a"]) for p in chunk])
        b = np.array([load_and_preprocess(p["img_b"]) for p in chunk])
        p_out = model.predict([a, b], verbose=0).flatten()
        probs.extend(p_out)
        labels.extend(int(p["label"]) for p in chunk)
    return np.array(labels), np.array(probs)

# ══════════════════════════════════════════════════════════════════════════════
# EXP 1 - SIAMESE DECISION THRESHOLD (tuned on val, applied to test)
# ══════════════════════════════════════════════════════════════════════════════
log("EXP 1: scoring val + test with siamese_best...")
siamese = tf.keras.models.load_model("outputs/models/siamese_best.keras",
                                     custom_objects=CUSTOM)
yv, pv = score_pairs(siamese, dataset.val_pairs)
yt, pt = score_pairs(siamese, dataset.test_pairs)

# sweep candidate thresholds on VAL only
cands = np.unique(np.round(pv, 4))
best_thr, best_f1v = 0.5, -1
for thr in cands:
    f1v = f1_score(yv, (pv >= thr).astype(int), zero_division=0)
    if f1v > best_f1v:
        best_f1v, best_thr = f1v, float(thr)

exp1 = {
    "default_0.5_test": prf(yt, (pt >= 0.5).astype(int)),
    "tuned_threshold":  round(best_thr, 4),
    "tuned_val_f1":     round(best_f1v, 4),
    "tuned_test":       prf(yt, (pt >= best_thr).astype(int)),
    "test_roc_auc":     round(roc_auc_score(yt, pt), 4),
}
RESULTS["exp1_siamese_threshold"] = exp1
log(f"EXP 1 done: default F1={exp1['default_0.5_test']['f1']}  "
    f"tuned thr={best_thr} → test F1={exp1['tuned_test']['f1']}")

# ══════════════════════════════════════════════════════════════════════════════
# EXP 2 - kNN DISTANCE THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════════
log("EXP 2: computing raw ResNet50 embeddings for test pairs...")
emb_model = get_embedding_model()

def embed_paths(paths):
    arrs = np.array([load_and_preprocess(p) for p in paths])
    return emb_model.predict(arrs, verbose=0)

emb_a = embed_paths([p["img_a"] for p in dataset.test_pairs])
emb_b = embed_paths([p["img_b"] for p in dataset.test_pairs])
y_knn = np.array([int(p["label"]) for p in dataset.test_pairs])
dists = np.linalg.norm(emb_a - emb_b, axis=1)

exp2 = {}
for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
    preds = (dists < thr).astype(int)
    exp2[f"dist_thr_{thr}"] = prf(y_knn, preds)
RESULTS["exp2_knn_distance_threshold"] = exp2
RESULTS["exp2_note"] = ("Baseline's k parameter is vestigial: predict_pair uses only "
                        "the Euclidean distance rule (face_recognition 0.6 convention). "
                        "Distance threshold is the operative hyperparameter.")
log(f"EXP 2 done: {json.dumps({k: v['f1'] for k, v in exp2.items()})}")

# ══════════════════════════════════════════════════════════════════════════════
# EXP 3 - AE PERCENTILE + COMPLIANT vs SYNTHETIC NON-COMPLIANT
# ══════════════════════════════════════════════════════════════════════════════
log("EXP 3: AE threshold percentiles vs synthetic grade drift...")
ae = tf.keras.models.load_model("outputs/models/autoencoder_best.keras",
                                custom_objects=CUSTOM)

def load_ae_img(path):
    return np.array(Image.open(path).convert("RGB").resize(AE_IMG_SIZE)) / 255.0

def distort(img_arr, mode):
    """Synthetic grade drift: hue shift / overexposure / desaturation."""
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    if mode == "hue_shift":
        hsv = np.array(img.convert("HSV"), dtype=np.uint8)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 256
        img = Image.fromarray(hsv, "HSV").convert("RGB")
    elif mode == "overexposure":
        img = ImageEnhance.Brightness(img).enhance(1.8)
    elif mode == "desaturation":
        img = ImageEnhance.Color(img).enhance(0.1)
    return np.array(img) / 255.0

compliant_paths = []
for subj in ["subject_A", "subject_B"]:
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        compliant_paths += list(Path(f"data/{subj}/styled").glob(ext))
compliant_paths = sorted(compliant_paths)

X_compliant = np.array([load_ae_img(p) for p in compliant_paths], dtype=np.float32)
mse_compliant = reconstruction_error(ae, X_compliant)

# distorted set: every 3rd compliant image, one distortion mode each (rotating)
modes = ["hue_shift", "overexposure", "desaturation"]
X_bad = np.array([distort(X_compliant[i], modes[i % 3])
                  for i in range(0, len(X_compliant), 3)], dtype=np.float32)
mse_bad = reconstruction_error(ae, X_bad)

exp3 = {
    "compliant_mean_mse":     round(float(mse_compliant.mean()), 6),
    "compliant_std_mse":      round(float(mse_compliant.std()), 6),
    "non_compliant_mean_mse": round(float(mse_bad.mean()), 6),
    "non_compliant_std_mse":  round(float(mse_bad.std()), 6),
    "n_compliant":            len(X_compliant),
    "n_non_compliant":        len(X_bad),
    "percentiles": {},
}
for pct in [90, 95, 99]:
    thr = float(np.percentile(mse_compliant, pct))
    false_alarm_rate = float((mse_compliant > thr).mean())   # compliant wrongly flagged
    drift_recall     = float((mse_bad > thr).mean())          # distorted correctly flagged
    exp3["percentiles"][f"pct_{pct}"] = {
        "threshold_mse":    round(thr, 6),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "drift_recall":     round(drift_recall, 4),
    }
RESULTS["exp3_ae_percentile"] = exp3
log(f"EXP 3 done: compliant MSE={exp3['compliant_mean_mse']}  "
    f"non-compliant MSE={exp3['non_compliant_mean_mse']}")

# ══════════════════════════════════════════════════════════════════════════════
# EXP 4 - EMBEDDING DIM 64 vs 128 vs 256 (retrains 64 + 256)
# ══════════════════════════════════════════════════════════════════════════════
from src.training.train_siamese import train as train_siamese, CONFIG

exp4 = {"dim_128": {"test": exp1["default_0.5_test"],
                    "test_roc_auc": exp1["test_roc_auc"],
                    "note": "existing siamese_best.keras"}}

for dim in [64, 256]:
    log(f"EXP 4: training embedding_dim={dim}...")
    cfg = dict(CONFIG)
    cfg["embedding_dim"]   = dim
    cfg["model_save_path"] = f"outputs/models/siamese_dim{dim}.keras"
    cfg["csv_log_path"]    = f"outputs/results/siamese_dim{dim}_log.csv"
    train_siamese(cfg)

    m = tf.keras.models.load_model(cfg["model_save_path"], custom_objects=CUSTOM)
    yd, pd_ = score_pairs(m, dataset.test_pairs)
    exp4[f"dim_{dim}"] = {
        "test":         prf(yd, (pd_ >= 0.5).astype(int)),
        "test_roc_auc": round(roc_auc_score(yd, pd_), 4),
    }
    del m
    log(f"EXP 4 dim={dim} done: F1={exp4[f'dim_{dim}']['test']['f1']}")

RESULTS["exp4_embedding_dim"] = exp4

# ── SAVE ──────────────────────────────────────────────────────────────────────
out = Path("outputs/results/parametrisation_experiments.json")
with open(out, "w") as f:
    json.dump(RESULTS, f, indent=2)
log(f"ALL EXPERIMENTS DONE → {out}")
print(json.dumps(RESULTS, indent=2))
