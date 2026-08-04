"""
baselines.py - the two opponents my siamese had to beat.

a fancy model only earns its place by beating something simpler. these two
define the floor: if the siamese cant beat a tracing-paper comparison and a
plain ruler, the deep learning was pointless.

opponent 1: pixel-difference detector (the tracing-paper kid)
  the EXISTING Mullay pipeline tool (CA1 §II). put one photo on top of the
  other, check how different each pixel is. fatal flaw: ANY angle change moves
  every pixel → "different person" even when it is the same woman. it compares
  pixels, not identity. (test recall 0.42 - missed most angle-changed pairs)

opponent 2: kNN on ResNet50 embeddings (the ruler with no student)
  measure euclidean distance between two fingerprints, call it same-person if
  closer than 0.6 - the face_recognition library's convention for NATURAL photos.
  my experiment 4.2 found that rule badly miscalibrated for AI-generated faces
  (they sit further apart in embedding space): recall 0.20 at 0.6, but F1=1.0 at 0.8.
  lesson: published conventions dont transfer blindly to synthetic imagery.
  note: the k parameter is vestigial - predict_pair never votes among k
  neighbours, it just measures one distance. kNN also never learns: fit() only
  STORES the data (lazy learner), and every dimension votes equally - noise as
  loud as jawline. a ruler, not a student (see my Section 1.3 note).
"""

import numpy as np
from PIL import Image
from pathlib import Path


# ── OPPONENT 1: PIXEL DIFFERENCE ─────────────────────────────────────────────

def pixel_difference_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    "mean pixel-level MSE between two same-size images. lower = more similar"
    #guard: are the pixels 0-255 or already 0-1? divide only if needed, so the score is always on the same scale
    a = img_a.astype(np.float32) / 255.0 if img_a.max() > 1 else img_a.astype(np.float32)
    b = img_b.astype(np.float32) / 255.0 if img_b.max() > 1 else img_b.astype(np.float32)
    #lay the tracing paper: subtract pixel by pixel, square, average into one number
    return float(np.mean((a - b) ** 2))


def pixel_diff_predict(img_a: np.ndarray, img_b: np.ndarray,
                       threshold: float = 0.05) -> dict:
    "classify a pair: pixel MSE below the threshold → same subject (close pixels = same person)"
    score = pixel_difference_score(img_a, img_b)
    return {
        "pixel_mse":     score,
        #1 = same subject, 0 = different
        "prediction":    1 if score < threshold else 0,
        "threshold":     threshold,
    }


# ── OPPONENT 2: kNN ON EMBEDDINGS ─────────────────────────────────────────────

class KNNEmbeddingClassifier:
    "the ruler: euclidean distance between two ResNet fingerprints vs a fixed 0.6 cutoff. nothing here ever learns"

    def __init__(self, k: int = 5, distance_threshold: float = 0.6):
        #k is kept for the classic kNN interface but predict_pair never uses it (vestigial - see experiment 4.2)
        self.k = k
        #the face_recognition library's 0.6 convention - calibrated on natural photos, not synthetic ones
        self.distance_threshold = distance_threshold
        self.train_embeddings = None
        self.train_labels     = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        "kNN is a lazy learner: 'fitting' just STORES the training data, nothing is learned"
        self.train_embeddings = embeddings.astype(np.float32)
        self.train_labels     = labels.astype(int)
        print(f"[knn] fitted on {len(labels)} training samples  (k={self.k})")

    #the ruler itself: straight-line distance between two fingerprints
    def _euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def predict_pair(self, emb_query: np.ndarray,
                     emb_reference: np.ndarray) -> dict:
        "one measurement: distance below the cutoff → same subject. every dimension votes equally, noise included"
        dist = self._euclidean(emb_query, emb_reference)
        return {
            "euclidean_distance": dist,
            "prediction":         1 if dist < self.distance_threshold else 0,
            "threshold":          self.distance_threshold,
        }

    def predict_batch(self, embeddings_a: np.ndarray,
                      embeddings_b: np.ndarray) -> np.ndarray:
        "measure every pair in the batch, return an array of 0/1 verdicts"
        preds = []
        for a, b in zip(embeddings_a, embeddings_b):
            preds.append(self.predict_pair(a, b)["prediction"])
        return np.array(preds)


# ── CONVENIENCE: RUN THE PIXEL-DIFF OPPONENT ON THE TEST SET ─────────────────

def evaluate_pixel_diff_baseline(test_pairs: list,
                                  threshold: float = 0.05) -> dict:
    "run the tracing-paper kid on every test pair, hand predictions + truth to evaluate.py"
    preds  = []
    labels = []
    for p in test_pairs:
        result = pixel_diff_predict(p["img_a_array"], p["img_b_array"], threshold)
        preds.append(result["prediction"])
        labels.append(p["label"])
    return {"predictions": np.array(preds), "labels": np.array(labels),
            "name": "pixel_difference", "threshold": threshold}
