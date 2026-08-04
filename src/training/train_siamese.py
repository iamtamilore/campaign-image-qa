"""
train_siamese.py - the school term: where the siamese network actually learns.

build = assemble the machine, compile = set the knobs, FIT = train. this file is the fit.

one epoch = one school day = the network sees all 420 training flashcards once.
after each day: a mini-quiz on the 90 validation cards (never learned from).
the 90 test cards stay SEALED until evaluate.py - that is the real exam.

KDD/CRISP-DM phase: Modelling (step 4 of 6).
dataset → preprocessing → pair construction MUST be done first.

run:
  python -m src.training.train_siamese

outputs:
  outputs/models/siamese_best.keras - the trained brain (best day, not last day)
  outputs/results/siamese_train_log.csv - the diary, one row per epoch
  outputs/results/siamese_train_summary.json - best score + epochs + time

my real result: early stop at epoch 11/50, val_roc_auc = 1.0000, 371s on CPU.
"""

import json, time
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                             ReduceLROnPlateau, CSVLogger)
    from src.models.siamese import build_siamese_model, compile_siamese
    from src.data.preprocessing import PairDataset
except ImportError:
    print("[warn] Install deps: pip install tensorflow")


# ── THE RECIPE CARD - every knob in one place, change one number without hunting ──
CONFIG = {
    #fingerprint size (FaceNet convention - experiment 4.4 retrained at 64 and 256 using this same dict)
    "embedding_dim":    128,
    #careful steps for a frozen backbone (see my rule: frozen → 1e-4, from-scratch → 1e-3)
    "learning_rate":    1e-4,
    #how many flashcards per weight adjustment
    "batch_size":       16,
    #max school days - early stopping usually ends it much sooner
    "epochs":           50,
    #coach: give up after 10 days without improvement
    "early_stop_patience": 10,
    #tutor: take smaller steps after 5 stuck days
    "reduce_lr_patience":   5,
    #smaller = half the current step size
    "reduce_lr_factor":     0.5,
    "manifest_path":    "data/pairs/pairs_manifest.json",
    "model_save_path":  "outputs/models/siamese_best.keras",
    "csv_log_path":     "outputs/results/siamese_train_log.csv",
}


def train(config: dict = CONFIG) -> dict:
    "full training run. returns dict with best val_roc_auc + epochs trained + time"
    # ── DATA - deal the flashcards into stacks, serve train + val ────────────
    dataset   = PairDataset(manifest_path=config["manifest_path"])
    train_ds  = dataset.as_tf_dataset("train", batch_size=config["batch_size"])
    val_ds    = dataset.as_tf_dataset("val",   batch_size=config["batch_size"])

    # ── MODEL - assemble the machine, then set the knobs ─────────────────────
    model = build_siamese_model(embedding_dim=config["embedding_dim"])
    model = compile_siamese(model, learning_rate=config["learning_rate"])
    model.summary()

    # ── THE FOUR SUPERVISORS (callbacks) ─────────────────────────────────────
    #make sure the save folders exist first
    Path(config["model_save_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["csv_log_path"]).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        #the coach: no val_roc_auc improvement for 10 days → school over,
        #and go back to the brain from the BEST day, not the last day (anti-overfitting)
        EarlyStopping(monitor="val_roc_auc", patience=config["early_stop_patience"],
                      restore_best_weights=True, mode="max"),
        #the photographer: snapshot the brain every time it beats its validation record
        ModelCheckpoint(config["model_save_path"], monitor="val_roc_auc",
                        save_best_only=True, mode="max"),
        #the patient tutor: stuck for 5 days → halve the step size (floor 1e-6)
        ReduceLROnPlateau(monitor="val_loss", patience=config["reduce_lr_patience"],
                          factor=config["reduce_lr_factor"], min_lr=1e-6),
        #the diary: write every day's scores to csv (evidence for the training-curves figure)
        CSVLogger(config["csv_log_path"]),
    ]

    # ── THE ACTUAL SCHOOL TERM ───────────────────────────────────────────────
    start = time.time()
    history = model.fit(
        train_ds,
        #the mini-quiz after every school day - never learned from
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=callbacks,
    )
    elapsed = time.time() - start

    # ── REPORT CARD ──────────────────────────────────────────────────────────
    best_roc = max(history.history.get("val_roc_auc", [0]))
    summary  = {
        "best_val_roc_auc":    best_roc,
        "epochs_trained":      len(history.history["loss"]),
        "training_time_s":     round(elapsed, 1),
        "config":              config,
    }

    #save the summary - this json is where the report's training numbers come from
    out_path = Path("outputs/results/siamese_train_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[train] done  best_val_roc_auc={best_roc:.4f}  time={elapsed:.0f}s")
    print(f"[train] model saved → {config['model_save_path']}")
    return summary


#run this file directly to retrain from scratch
if __name__ == "__main__":
    train()
