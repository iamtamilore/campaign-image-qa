import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

N_WARMUP = 3      # first calls include graph tracing, never time those
N_RUNS = 30


def percentile_stats(times_ms):
    a = np.array(times_ms)
    return {
        "mean_ms": round(float(a.mean()), 2),
        "std_ms": round(float(a.std()), 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "min_ms": round(float(a.min()), 2),
        "max_ms": round(float(a.max()), 2),
    }


def main():
    import tensorflow as tf
    from src.models.backbone import L2Normalize
    from src.models.siamese import L1Distance
    from src.data.preprocessing import load_and_preprocess

    custom = {"L1Distance": L1Distance, "L2Normalize": L2Normalize}
    siamese = tf.keras.models.load_model(
        ROOT / "outputs/models/siamese_best.keras", custom_objects=custom, compile=False)
    autoenc = tf.keras.models.load_model(
        ROOT / "outputs/models/autoencoder_best.keras", compile=False)

    pairs = json.loads((ROOT / "data/pairs/pairs_manifest.json").read_text())
    sample = pairs[0]
    img_a, img_b = ROOT / sample["img_a"], ROOT / sample["img_b"]

    results = {"device": "CPU" if not tf.config.list_physical_devices("GPU") else "GPU",
               "n_runs": N_RUNS}

    # ---- stage 1: disk -> preprocessed tensor -------------------------------
    for _ in range(N_WARMUP):
        load_and_preprocess(str(img_a))
    t = []
    for _ in range(N_RUNS):
        s = time.perf_counter()
        load_and_preprocess(str(img_a))
        t.append((time.perf_counter() - s) * 1000)
    results["preprocess_per_image"] = percentile_stats(t)

    a = np.expand_dims(load_and_preprocess(str(img_a)), 0)
    b = np.expand_dims(load_and_preprocess(str(img_b)), 0)

    # ---- stage 2: siamese identity verdict on one pair ----------------------
    for _ in range(N_WARMUP):
        siamese.predict([a, b], verbose=0)
    t = []
    for _ in range(N_RUNS):
        s = time.perf_counter()
        siamese.predict([a, b], verbose=0)
        t.append((time.perf_counter() - s) * 1000)
    results["siamese_pair_inference"] = percentile_stats(t)

    # ---- stage 3: autoencoder grade verdict on one image --------------------
    ae_in = np.expand_dims(
        np.array(Image.open(img_a).convert("RGB").resize((128, 128))) / 255.0, 0)
    for _ in range(N_WARMUP):
        autoenc.predict(ae_in, verbose=0)
    t = []
    for _ in range(N_RUNS):
        s = time.perf_counter()
        autoenc.predict(ae_in, verbose=0)
        t.append((time.perf_counter() - s) * 1000)
    results["autoencoder_inference"] = percentile_stats(t)

    gate = (results["preprocess_per_image"]["mean_ms"]
            + results["siamese_pair_inference"]["mean_ms"]
            + results["autoencoder_inference"]["mean_ms"])
    results["end_to_end_gate_mean_ms"] = round(gate, 2)
    results["throughput_images_per_min"] = round(60000.0 / gate, 1)
    results["note"] = (
        "Latency of the deployed QA gate on one new image: preprocess, Siamese "
        "identity check against the hero reference, autoencoder grade check. "
        f"{N_WARMUP} warm-up calls discarded (graph tracing), {N_RUNS} timed runs.")

    out = ROOT / "outputs/results/latency.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"device: {results['device']}   runs: {N_RUNS}")
    for k in ("preprocess_per_image", "siamese_pair_inference", "autoencoder_inference"):
        s = results[k]
        print(f"  {k:28s} mean {s['mean_ms']:8.2f} ms   p95 {s['p95_ms']:8.2f} ms")
    print(f"  {'END-TO-END GATE':28s} mean {gate:8.2f} ms "
          f"({results['throughput_images_per_min']} images/min)")
    print(f"saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
