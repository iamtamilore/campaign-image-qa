"""
backbone.py - the fingerprint machine: frozen ResNet50, photo in → 128 numbers out.

why ResNet50:
 - pre-trained on ImageNet (1.2M natural images) - i hire the expert, i dont train an intern
 - frozen so it cannot overfit on my small dataset (106 images)
 - transfer learning rationale: CA1 §I + Koch et al. (2015) few-shot framing

key design choice from CA1:
  the domain gap between natural faces and AI-generated fashion images means
  face_recognition (dlib HOG) fails. ResNet50 features capture texture, shape,
  AND colour, making them more robust to synthetic faces.
"""

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Layer
except ImportError:
    print("[warn] TensorFlow not installed - pip install tensorflow")


#squeeze the 2048-number report down to 128 - FaceNet convention (Schroff et al., 2015)
#experiment 4 later showed 64/128/256 all reach ROC-AUC 1.0, so this choice was safe
EMBEDDING_DIM = 128


class L2Normalize(tf.keras.layers.Layer):
    "stretch/shrink every fingerprint to length exactly 1, so only its DIRECTION matters - makes distances fair. custom layer because Lambda layers dont save/load properly in Keras 3"
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

    #keras calls this itself when saving/loading the model - do not delete
    def compute_output_shape(self, input_shape):
        return input_shape


#build the shared fingerprint machine used by BOTH arms of the siamese net
def build_embedding_network(input_shape: tuple = (224, 224, 3),
                             embedding_dim: int = EMBEDDING_DIM,
                             trainable_backbone: bool = False) -> "Model":
    "ResNet50 (ImageNet weights, frozen) → GlobalAvgPool → Dense(128) → L2-norm. maps (224,224,3) → (128,)"
    #hire the expert WITH her ImageNet experience, but cut off her old answer sheet
    #(include_top=False drops the is-it-a-cat/dog/car layer - useless to me, i keep the eyes)
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    #the freeze: not allowed to re-learn on my job - 106 images would wreck a 1.2M-image education
    base.trainable = trainable_backbone

    x = base.output
    #average her raw notes into one 2048-number report per photo
    x = GlobalAveragePooling2D()(x)       # (None, 2048)
    #the funnel: squeeze 2048 down to the 128-number summary
    x = Dense(embedding_dim)(x)           # (None, 128)
    #normalise so every fingerprint has length 1
    x = L2Normalize()(x)

    return Model(inputs=base.input, outputs=x, name="embedding_network")


#convenience shortcut - same machine, default settings
def get_embedding_model(embedding_dim: int = EMBEDDING_DIM) -> "Model":
    "returns the shared embedding model"
    return build_embedding_network(embedding_dim=embedding_dim)
