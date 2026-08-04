"""
run_advanced_experiments.py - High-H1 evaluation depth (Rubric: Methodology 30% + Evaluation 20%).

Four experiments, all local/GPU, no Kie.ai credits. Saves incrementally so a
failure in one experiment never loses the others.

  EXP 1 - SAMPLING METHODS: repeated stratified splits (5 seeds = Monte-Carlo CV).
          Reports mean±std of P/R/F1/ROC-AUC → "rich understanding of performance".
  EXP 2 - DISTANCE METRIC: L1 vs Euclidean vs cosine head (resolves proposal-vs-build
          mismatch: proposal promised Euclidean, build used L1).
  EXP 3 - ABLATION LADDER + McNEMAR: raw-pixel distance vs embedding distance vs
          Siamese (threshold-free ROC-AUC ladder) + McNemar's test (proposal ref [9]).
  EXP 4 - NEGATIVE SAMPLING: hard (angle-matched cross-subject) vs random negatives
          (proves the "matched to prevent style shortcuts" claim).

Output: outputs/results/advanced_experiments.json
"""

import json, time, warnings, random
import numpy as np
from pathlib import Path

# ── run-from-anywhere shim (dir reorg 2026-07-07): anchor CWD + imports to project root ──
import os as _os, sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

from PIL import Image

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score)
from scipy.stats import chi2

from src.data.preprocessing import PairDataset, load_and_preprocess
from src.models.backbone import build_embedding_network, L2Normalize
from src.models.siamese import build_siamese_model, compile_siamese, L1Distance

RESULTS = {}
OUT = Path("outputs/results/advanced_experiments.json")
t0 = time.time()

def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
def save(): OUT.write_text(json.dumps(RESULTS, indent=2))

def prf(y, p):
    return {"precision": round(precision_score(y, p, zero_division=0), 4),
            "recall":    round(recall_score(y, p, zero_division=0), 4),
            "f1":        round(f1_score(y, p, zero_division=0), 4)}

def score_pairs(model, pairs, batch=16):
    probs, labels = [], []
    for i in range(0, len(pairs), batch):
        c = pairs[i:i+batch]
        a = np.array([load_and_preprocess(p["img_a"]) for p in c])
        b = np.array([load_and_preprocess(p["img_b"]) for p in c])
        probs.extend(model.predict([a, b], verbose=0).flatten())
        labels.extend(int(p["label"]) for p in c)
    return np.array(labels), np.array(probs)

def make_tf_ds(pairs, batch=16, augment=False):
    from src.data.preprocessing import load_pair, IMG_SIZE
    def gen():
        for p in pairs:
            a, b, l = load_pair(p, augment=augment)
            yield (a, b), l
    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        (tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int32)))
    return ds.shuffle(200).batch(batch).prefetch(tf.data.AUTOTUNE)

def train_eval(train_p, val_p, test_p, model, epochs=40):
    model = compile_siamese(model)
    model.fit(make_tf_ds(train_p, augment=True),
              validation_data=make_tf_ds(val_p),
              epochs=epochs, verbose=0,
              callbacks=[EarlyStopping(monitor="val_roc_auc", patience=8,
                                       restore_best_weights=True, mode="max")])
    y, p = score_pairs(model, test_p)
    m = prf(y, (p >= 0.5).astype(int))
    m["roc_auc"] = round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None
    return m, y, p

# ══════════════════════════════════════════════════════════════════════════════
# EXP 1 - SAMPLING METHODS: repeated stratified splits (Monte-Carlo CV)
# ══════════════════════════════════════════════════════════════════════════════
def exp1():
    log("EXP 1: repeated stratified splits across 5 seeds...")
    seeds = [42, 7, 123, 2024, 99]
    runs = []
    for s in seeds:
        ds = PairDataset(seed=s)
        m, _, _ = train_eval(ds.train_pairs, ds.val_pairs, ds.test_pairs,
                             build_siamese_model())
        runs.append(m)
        log(f"  seed {s}: F1={m['f1']} AUC={m['roc_auc']}")
    def agg(key):
        vals = [r[key] for r in runs if r[key] is not None]
        return {"mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4)}
    RESULTS["exp1_sampling_montecarlo_cv"] = {
        "seeds": seeds, "per_run": runs,
        "precision": agg("precision"), "recall": agg("recall"),
        "f1": agg("f1"), "roc_auc": agg("roc_auc"),
        "note": "5 independent stratified 70/15/15 resamples (Monte-Carlo CV). "
                "Mean±std quantifies performance stability across sampling."}
    save(); log("EXP 1 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# EXP 2 - DISTANCE METRIC: L1 vs Euclidean vs cosine
# ══════════════════════════════════════════════════════════════════════════════
class L2Distance(tf.keras.layers.Layer):
    def call(self, t):
        return tf.sqrt(tf.reduce_sum(tf.square(t[0]-t[1]), axis=1, keepdims=True)
                       + tf.keras.backend.epsilon())
    def compute_output_shape(self, s): return (s[0][0], 1)

class CosineDistance(tf.keras.layers.Layer):
    def call(self, t):  # embeddings are L2-normalised → cosine = dot product
        return 1.0 - tf.reduce_sum(t[0]*t[1], axis=1, keepdims=True)
    def compute_output_shape(self, s): return (s[0][0], 1)

def build_metric_siamese(distance_layer):
    net = build_embedding_network(embedding_dim=128)
    ia, ib = Input((224,224,3), name="input_a"), Input((224,224,3), name="input_b")
    ea, eb = net(ia), net(ib)
    d = distance_layer([ea, eb])
    out = Dense(1, activation="sigmoid")(d)
    return Model([ia, ib], out)

def exp2():
    log("EXP 2: distance metric sweep...")
    ds = PairDataset(seed=42)
    variants = {"L1_current": L1Distance(), "Euclidean_proposal": L2Distance(),
                "cosine": CosineDistance()}
    res = {}
    for name, layer in variants.items():
        m, _, _ = train_eval(ds.train_pairs, ds.val_pairs, ds.test_pairs,
                             build_metric_siamese(layer))
        res[name] = m
        log(f"  {name}: F1={m['f1']} AUC={m['roc_auc']}")
    RESULTS["exp2_distance_metric"] = {
        **res,
        "note": "Proposal specified Euclidean; implementation used L1. All three "
                "compared here to justify the head choice empirically."}
    save(); log("EXP 2 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# EXP 3 - ABLATION LADDER (raw pixels → embeddings → Siamese) + McNemar
# ══════════════════════════════════════════════════════════════════════════════
def flat_pixel_dist(pairs, size=(64,64)):
    d = []
    for p in pairs:
        a = np.array(Image.open(p["img_a"]).convert("RGB").resize(size), np.float32)/255
        b = np.array(Image.open(p["img_b"]).convert("RGB").resize(size), np.float32)/255
        d.append(np.linalg.norm(a.flatten()-b.flatten()))
    return np.array(d)

def exp3():
    log("EXP 3: ablation ladder + McNemar...")
    ds = PairDataset(seed=42)
    test = ds.test_pairs
    y = np.array([int(p["label"]) for p in test])

    # rung 1 - raw pixel distance (lower dist = more similar → score = -dist)
    pix_d = flat_pixel_dist(test)
    auc_pixel = round(roc_auc_score(y, -pix_d), 4)

    # rung 2 - raw ResNet50 embedding distance (no metric learning)
    from src.models.backbone import get_embedding_model
    emb = get_embedding_model()
    def embed(paths):
        return emb.predict(np.array([load_and_preprocess(p) for p in paths]), verbose=0)
    ea = embed([p["img_a"] for p in test]); eb = embed([p["img_b"] for p in test])
    emb_d = np.linalg.norm(ea-eb, axis=1)
    auc_emb = round(roc_auc_score(y, -emb_d), 4)

    # rung 3 - Siamese (frozen backbone, the built model)
    sm = tf.keras.models.load_model("outputs/models/siamese_best.keras",
            custom_objects={"L2Normalize": L2Normalize, "L1Distance": L1Distance})
    _, prob_siam = score_pairs(sm, test)
    auc_siam = round(roc_auc_score(y, prob_siam), 4)
    pred_siam = (prob_siam >= 0.5).astype(int)

    # rung 4 - Siamese with fine-tuned (unfrozen) backbone
    net_ft = build_embedding_network(embedding_dim=128, trainable_backbone=True)
    ia, ib = Input((224,224,3), name="input_a"), Input((224,224,3), name="input_b")
    d = L1Distance()([net_ft(ia), net_ft(ib)])
    ft = Model([ia, ib], Dense(1, activation="sigmoid")(d))
    ft = compile_siamese(ft, learning_rate=1e-5)  # low lr for fine-tuning
    ft.fit(make_tf_ds(ds.train_pairs, augment=True),
           validation_data=make_tf_ds(ds.val_pairs), epochs=25, verbose=0,
           callbacks=[EarlyStopping(monitor="val_roc_auc", patience=6,
                                    restore_best_weights=True, mode="max")])
    y_ft, prob_ft = score_pairs(ft, test)
    auc_ft = round(roc_auc_score(y_ft, prob_ft), 4)

    # McNemar: Siamese vs pixel-diff baseline (threshold pixel dist at F1-best on val)
    val = ds.val_pairs
    yv = np.array([int(p["label"]) for p in val])
    pv = flat_pixel_dist(val)
    best_thr, best = 0, -1
    for thr in np.percentile(pv, np.arange(5, 96, 5)):
        f = f1_score(yv, (pv < thr).astype(int), zero_division=0)
        if f > best: best, best_thr = f, thr
    pred_pixel = (pix_d < best_thr).astype(int)

    siam_correct  = (pred_siam == y)
    pixel_correct = (pred_pixel == y)
    b = int(np.sum(siam_correct & ~pixel_correct))   # Siamese right, pixel wrong
    c = int(np.sum(~siam_correct & pixel_correct))   # Siamese wrong, pixel right
    stat = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    pval = float(chi2.sf(stat, df=1)) if (b + c) > 0 else 1.0

    RESULTS["exp3_ablation_and_mcnemar"] = {
        "ablation_ladder_roc_auc": {
            "1_raw_pixel_distance":      auc_pixel,
            "2_resnet50_embedding_dist": auc_emb,
            "3_siamese_frozen":          auc_siam,
            "4_siamese_finetuned":       auc_ft},
        "mcnemar_siamese_vs_pixeldiff": {
            "b_siamese_right_pixel_wrong": b,
            "c_siamese_wrong_pixel_right": c,
            "statistic": round(stat, 4), "p_value": round(pval, 6),
            "significant_at_0.05": pval < 0.05},
        "note": "Threshold-free ROC-AUC isolates each added component. McNemar tests "
                "whether the Siamese error pattern differs significantly from pixel-diff."}
    save(); log(f"EXP 3 saved. ladder AUC {auc_pixel}->{auc_emb}->{auc_siam}->{auc_ft}, "
                f"McNemar p={pval:.4g}")

# ══════════════════════════════════════════════════════════════════════════════
# EXP 4 - NEGATIVE SAMPLING: hard (angle-matched) vs random
# ══════════════════════════════════════════════════════════════════════════════
def build_pairs_custom(strategy, n_per_class=300, seed=42):
    """Positives = same subject. Negatives = cross-subject, either angle-matched
       (hard) or random."""
    rng = random.Random(seed)
    def angle_of(path):
        for a in ["3qleft","3qright","profileleft","profileright","back","front"]:
            if a in path: return a
        return "other"
    subj = {}
    for s in ["subject_A","subject_B"]:
        styled = sorted(str(p) for p in Path(f"data/{s}/styled").glob("*.png"))
        heroes = sorted(str(p) for p in Path(f"data/{s}/hero_ref").glob("*.png"))
        subj[s] = {"styled": styled, "heroes": heroes}
    # positives
    pos = []
    for s in ["subject_A","subject_B"]:
        for h in subj[s]["heroes"]:
            for st in subj[s]["styled"]:
                pos.append({"img_a": h, "img_b": st, "label": 1})
    # negatives
    neg = []
    for s, o in [("subject_A","subject_B"),("subject_B","subject_A")]:
        for h in subj[s]["heroes"]:
            for st in subj[o]["styled"]:
                if strategy == "hard" and angle_of(h) != angle_of(st):
                    continue
                neg.append({"img_a": h, "img_b": st, "label": 0})
    pos = rng.sample(pos, min(n_per_class, len(pos)))
    neg = rng.sample(neg, min(n_per_class, len(neg)))
    allp = pos + neg; rng.shuffle(allp)
    # stratified 70/15/15
    def split(lst):
        n = len(lst); return lst[:int(.7*n)], lst[int(.7*n):int(.85*n)], lst[int(.85*n):]
    p_tr,p_va,p_te = split([p for p in allp if p["label"]==1])
    n_tr,n_va,n_te = split([p for p in allp if p["label"]==0])
    return p_tr+n_tr, p_va+n_va, p_te+n_te

def exp4():
    log("EXP 4: hard vs random negative sampling...")
    res = {}
    for strat in ["hard", "random"]:
        tr, va, te = build_pairs_custom(strat)
        m, _, _ = train_eval(tr, va, te, build_siamese_model())
        res[strat] = {**m, "n_train": len(tr), "n_test": len(te)}
        log(f"  {strat} negatives: F1={m['f1']} AUC={m['roc_auc']}")
    RESULTS["exp4_negative_sampling"] = {
        **res,
        "note": "Hard = angle-matched cross-subject (model cannot cheat on pose). "
                "Random = any cross-subject. Tests the proposal's 'matched to prevent "
                "style shortcuts' design claim."}
    save(); log("EXP 4 saved.")

# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("GPU:", tf.config.list_physical_devices('GPU'))
    for fn in [exp1, exp2, exp3, exp4]:
        try:
            fn()
        except Exception as e:
            import traceback
            log(f"!! {fn.__name__} FAILED: {e}")
            traceback.print_exc()
            RESULTS[f"{fn.__name__}_error"] = str(e)
            save()
    log(f"ALL DONE → {OUT}")
    print(json.dumps(RESULTS, indent=2))
