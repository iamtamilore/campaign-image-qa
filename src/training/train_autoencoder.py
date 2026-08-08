import json, time, numpy as np
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from src.models.autoencoder import (build_autoencoder, compile_autoencoder,
                                         reconstruction_error, AE_IMG_SIZE)
except ImportError:
    print("[warn] Install deps: pip install tensorflow pillow numpy")


# ── THE RECIPE CARD ───────────────────────────────────────────────────────────
CONFIG = {
    #the intern's whole memory: 64 numbers (the bottleneck)
    "latent_dim":           64,
    #from-scratch training → bigger steps than the frozen-backbone 1e-4
    "learning_rate":        1e-3,
    #small batches, the compliant set is only 106 images
    "batch_size":           8,
    #up to 100 school days, cheap model, patience is affordable
    "epochs":               100,
    "early_stop_patience":  15,
    #the verdict line: worse than 95% of normal redraw errors = suspicious
    "threshold_percentile": 95,
    "model_save_path":      "outputs/models/autoencoder_best.keras",
    "threshold_save_path":  "outputs/results/ae_threshold.json",
    "csv_log_path":         "outputs/results/ae_train_log.csv",
}


def load_compliant_images(data_dir: str = "data") -> np.ndarray:
    "load all known-good images from both subjects (styled + hero_ref), 'here is what normal looks like'"
    data_path = Path(data_dir)
    paths = []
    for subj in ["subject_A", "subject_B"]:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            paths += list((data_path / subj / "styled").glob(ext))
            paths += list((data_path / subj / "hero_ref").glob(ext))

    imgs = []
    for p in paths:
        #resize to the AE's 128x128 tube size
        img = load_img(str(p), target_size=AE_IMG_SIZE)
        #divide by 255 (NOT ImageNet preprocessing): the decoder ends in a sigmoid
        #that can only output 0-1, so the input must live in 0-1 too
        arr = img_to_array(img) / 255.0
        imgs.append(arr)

    print(f"[ae] loaded {len(imgs)} compliant images from {data_dir}")
    return np.array(imgs, dtype=np.float32)   # (N, 128, 128, 3)


def train(config: dict = CONFIG) -> dict:
    "full AE training run: school, then draw the verdict line. returns the threshold stats dict"
    images = load_compliant_images()

    #safety net: cant train a redraw intern on almost nothing
    if len(images) < 4:
        raise ValueError(
            f"Only {len(images)} images found. Add Mullay images to data/subject_A/styled/ "
            "and data/subject_B/styled/ before training."
        )

    #80% train / 20% val, no test stack here, the real test came later
    #with synthetic bad images (hue shift, overexposure, experiment 4.3)
    n_val   = max(1, int(0.2 * len(images)))
    idx     = np.random.permutation(len(images))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    X_train = images[tr_idx]
    X_val   = images[val_idx]

    #assemble the squeeze-and-redraw machine, then set the knobs
    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=(AE_IMG_SIZE[0], AE_IMG_SIZE[1], 3),
        latent_dim=config["latent_dim"]
    )
    autoencoder = compile_autoencoder(autoencoder, config["learning_rate"])
    autoencoder.summary()

    Path(config["model_save_path"]).parent.mkdir(parents=True, exist_ok=True)

    # ── THREE SUPERVISORS (no ReduceLROnPlateau, 1e-3 converges fine here) ──
    callbacks = [
        #the coach: monitors val_loss (no classes to rank here, the only score is redraw error)
        EarlyStopping(monitor="val_loss", patience=config["early_stop_patience"],
                      restore_best_weights=True),
        #the photographer: snapshot the best brain
        ModelCheckpoint(config["model_save_path"], monitor="val_loss", save_best_only=True),
        #the diary
        CSVLogger(config["csv_log_path"]),
    ]

    # ── SCHOOL, the strangest line in the project: input = target ──────────
    start = time.time()
    history = autoencoder.fit(
        X_train, X_train,           #redraw homework: the answer sheet IS the photo
        validation_data=(X_val, X_val),
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        callbacks=callbacks,
    )
    elapsed = time.time() - start

    # ── DRAW THE VERDICT LINE ────────────────────────────────────────────────
    #measure the intern's redraw error on every good training image
    train_mse = reconstruction_error(autoencoder, X_train)
    #the line goes at the 95th percentile of his normal errors
    threshold = float(np.percentile(train_mse, config["threshold_percentile"]))

    threshold_data = {
        "threshold_mse":            threshold,
        "percentile":               config["threshold_percentile"],
        "train_mse_mean":           float(train_mse.mean()),
        "train_mse_std":            float(train_mse.std()),
        "n_training_images":        len(X_train),
        "final_val_loss":           float(history.history["val_loss"][-1]),
        "training_time_s":          round(elapsed, 1),
    }

    #save the line + its stats, this json is where the report's AE numbers come from
    Path(config["threshold_save_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(config["threshold_save_path"], "w") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"\n[ae] done  threshold_MSE={threshold:.6f}  time={elapsed:.0f}s")
    print(f"[ae] model saved → {config['model_save_path']}")
    print(f"[ae] threshold saved → {config['threshold_save_path']}")
    return threshold_data


#run this file directly to retrain from scratch
if __name__ == "__main__":
    train()
