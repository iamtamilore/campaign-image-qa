import json
import numpy as np
from pathlib import Path

#tensorflow is heavy, so we check it loads and warn instead of crashing the import
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[warn] TensorFlow not installed, install with: pip install tensorflow")


#the tube size, ResNet50 only accepts this
IMG_SIZE = (224, 224)


#one photo from disk → machine-ready array
def load_and_preprocess(image_path: str, augment: bool = False) -> "np.ndarray":
    "load image, resize to 224x224, apply the ImageNet colour shift. returns (224, 224, 3) float32"
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required. pip install tensorflow")

    #open the image and resize it to the tube size in one go
    img   = load_img(image_path, target_size=IMG_SIZE)
    #turn the picture into a grid of numbers (224, 224, 3) float32
    arr   = img_to_array(img)
    #the dilution: shift colours by the same mean values ImageNet training used
    arr   = preprocess_input(arr)

    #only the train stack ever gets augmented
    if augment:
        arr = _augment(arr)

    return arr


#the coin-flip mirror + small brightness nudge (train only)
def _augment(arr: "np.ndarray") -> "np.ndarray":
    "light augmentation for training only. NEVER on val/test, that would be data leakage"
    #coin flip: mirror the image left-right half the time
    if np.random.rand() > 0.5:
        arr = np.fliplr(arr)
    #brightness nudge of max ±10%, small on purpose, must not change the grade signature
    delta = np.random.uniform(-0.1, 0.1)
    #clip keeps the numbers inside the range ResNet expects (ImageNet space is centred)
    arr   = np.clip(arr + delta * 255, -128, 128)
    return arr


#load one flashcard: both photos + the answer on the back
def load_pair(pair: dict, augment: bool = False) -> tuple:
    "load a single pair dict from the manifest, returns (img_a_array, img_b_array, label)"
    img_a = load_and_preprocess(pair["img_a"], augment=augment)
    img_b = load_and_preprocess(pair["img_b"], augment=augment)
    #the answer: 1 = same person, 0 = different
    label = int(pair["label"])
    return img_a, img_b, label


class PairDataset:
    "reads pairs_manifest.json and deals the flashcards into train/val/test stacks, keeping pos/neg balance in each"

    def __init__(self, manifest_path: str = "data/pairs/pairs_manifest.json",
                 val_split: float = 0.15, test_split: float = 0.15,
                 seed: int = 42):
        #read the 600 flashcards
        with open(manifest_path) as f:
            self.pairs = json.load(f)

        #split the deck into red and black first (stratified: keep pos/neg ratio in each stack)
        positives = [p for p in self.pairs if p["label"] == 1]
        negatives = [p for p in self.pairs if p["label"] == 0]

        #deal each colour separately, then combine, every stack ends up 50/50
        self.train_pairs, self.val_pairs, self.test_pairs = \
            self._split(positives, negatives, val_split, test_split, seed)

        print(f"[dataset] train={len(self.train_pairs)}  val={len(self.val_pairs)}"
              f"  test={len(self.test_pairs)}")

    #the dealer: shuffle with seed 42 so the deal is identical every run
    def _split(self, pos, neg, val_frac, test_frac, seed):
        rng = np.random.default_rng(seed)
        def _s(lst):
            #shuffle the card positions
            idx = rng.permutation(len(lst))
            #how many cards go to test and val
            n_test = int(len(lst) * test_frac)
            n_val  = int(len(lst) * val_frac)
            #first cards → test, next → val, the rest → train
            return ([lst[i] for i in idx[n_test + n_val:]],   #train
                    [lst[i] for i in idx[n_test:n_test + n_val]],   #val
                    [lst[i] for i in idx[:n_test]])   #test
        #deal positives and negatives separately, then glue the stacks together
        ptr, pvr, pte = _s(pos)
        ntr, nvr, nte = _s(neg)
        return ptr + ntr, pvr + nvr, pte + nte

    #serve one stack to tensorflow in batches of 16
    def as_tf_dataset(self, split: str = "train", batch_size: int = 16) -> "tf.data.Dataset":
        "return a tf.data.Dataset of ((img_a, img_b), label) batches for the chosen stack"
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")

        #pick which stack we are serving
        pairs  = {"train": self.train_pairs, "val": self.val_pairs,
                  "test": self.test_pairs}[split]
        #ONLY the train stack gets the mirror/brightness treatment
        augment = (split == "train")

        #a generator: hands tensorflow one loaded flashcard at a time
        def _gen():
            for p in pairs:
                a, b, lbl = load_pair(p, augment=augment)
                yield (a, b), lbl

        #tell tensorflow the exact shape and type of what the generator serves
        ds = tf.data.Dataset.from_generator(
            _gen,
            output_signature=(
                (tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32)),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
        )
        #seed=42 so batch order is the same every run (reproducible metrics)
        return ds.shuffle(200, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)
