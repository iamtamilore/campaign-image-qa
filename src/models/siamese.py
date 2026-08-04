"""
siamese.py - the twins with one brain: my network for same-person-or-not verification.

"siamese" sounds like two networks but it is ONE fingerprint machine used TWICE - 
same doctor, same eyes, holding one photo in each hand. if two different doctors
examined the photos, their notes would not be comparable.

architecture:
  photo A ─→ ┐
             ├─ SAME frozen ResNet50 arm (shared weights) → two 128-number fingerprints
  photo B ─→ ┘
  → L1 distance (the 128 differences) → Dense(1, sigmoid) → probability of same person [0,1]

loss: binary cross-entropy (contrastive loss variant discussed in CA1 §IV)
threshold: 0.5 by default (experiment 4.1 later tuned it to 0.4821 on validation → F1 1.0)

references (CA1 §IV):
  [1] Schroff et al., FaceNet (2015)
  [2] Koch et al., Siamese one-shot (2015)
  [6] Chopra et al., contrastive loss (2005)
"""

#fallback if the import below fails; normally overwritten by the backbone import
EMBEDDING_DIM = 128

try:
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Dense
    from src.models.backbone import build_embedding_network, EMBEDDING_DIM
except ImportError:
    print("[warn] TensorFlow not installed - pip install tensorflow")


class L1Distance(tf.keras.layers.Layer):
    "subtract the two fingerprints and drop the minus signs: |a - b| per dimension = 128 difference values. custom layer for Keras 3 save/load"
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    #keras calls this itself when saving/loading the model - do not delete
    def compute_output_shape(self, input_shape):
        return input_shape[0]


def build_siamese_model(input_shape: tuple = (224, 224, 3),
                        embedding_dim: int = EMBEDDING_DIM) -> "Model":
    "two photos in → probability they show the same subject [0,1]"
    #build the fingerprint machine ONCE - this is the whole siamese trick
    embedding_net = build_embedding_network(embedding_dim=embedding_dim)

    #two doors for the two photos
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")

    #same machine called twice = shared weights, literally the same brain reading both
    emb_a = embedding_net(input_a)    # (batch, 128)
    emb_b = embedding_net(input_b)

    #the 128 differences: same person → mostly tiny, different people → many big
    l1_dist = L1Distance(name="l1_distance")([emb_a, emb_b])

    #the judgement - the ONLY layer here that learns: WHICH differences matter.
    #sigmoid squashes the verdict to a 0-1 probability
    out = Dense(1, activation="sigmoid", name="same_subject_prob")(l1_dist)

    model = Model(inputs=[input_a, input_b], outputs=out, name="siamese_network")
    return model


def compile_siamese(model: "Model", learning_rate: float = 1e-4) -> "Model":
    "set the training rules: Adam tutor + binary cross-entropy punishment + the rubric scorecard"
    model.compile(
        #the tutor: adjusts weights after each mistake, in small careful steps (1e-4)
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        #the punishment rule: confidently wrong = big penalty
        loss="binary_crossentropy",
        #the scorecard printed every epoch (CA2 rubric: Evaluation 20%)
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ]
    )
    return model


def predict_pair(model: "Model", img_a: "np.ndarray", img_b: "np.ndarray",
                 threshold: float = 0.5) -> dict:
    "the deployment doorbell: one pair in → score + verdict out. used by --phase qa on new images"
    import numpy as np
    #the model expects batches, so wrap each single photo in a batch of 1
    a = np.expand_dims(img_a, 0)
    b = np.expand_dims(img_b, 0)
    #run the network, unwrap the single number out of the batch
    score = float(model.predict([a, b], verbose=0)[0][0])
    return {
        "score": score,
        #score above the threshold → same subject
        "same_subject": score >= threshold,
        "threshold_used": threshold,
    }
