try:
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (Conv2D, Conv2DTranspose,
                                          MaxPooling2D, UpSampling2D,
                                          Reshape, Flatten, Dense)
except ImportError:
    print("[warn] TensorFlow not installed, pip install tensorflow")

import numpy as np

#smaller tube size than the siamese, the AE doesnt need ResNet's 224
AE_IMG_SIZE = (128, 128)


def build_autoencoder(input_shape: tuple = (128, 128, 3),
                      latent_dim: int = 64) -> tuple:
    "build the squeeze-and-redraw machine. returns (autoencoder, encoder, decoder), three keras models"
    # ── ENCODER, the squeeze: 49,152 numbers down to 64 ────────────────────
    inp = Input(shape=input_shape, name="ae_input")
    #each Conv2D looks for patterns, each MaxPooling halves the picture size
    x   = Conv2D(32, 3, activation="relu", padding="same")(inp)
    x   = MaxPooling2D(2)(x)          # 128 → 64
    x   = Conv2D(64, 3, activation="relu", padding="same")(x)
    x   = MaxPooling2D(2)(x)          # 64 → 32
    x   = Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = MaxPooling2D(2)(x)          # 32 → 16
    #unroll the little 16x16 grid into one long line of numbers
    x   = Flatten()(x)
    #the bottleneck itself: everything the intern is allowed to remember = 64 numbers
    latent = Dense(latent_dim, name="latent")(x)

    encoder = Model(inp, latent, name="encoder")

    # ── DECODER, the redraw: 64 numbers back up to a full image ────────────
    dec_inp = Input(shape=(latent_dim,), name="decoder_input")
    #expand the 64 memory numbers back into a small 16x16 grid
    x   = Dense(16 * 16 * 128, activation="relu")(dec_inp)
    x   = Reshape((16, 16, 128))(x)
    #mirror of the encoder: Conv2DTranspose redraws patterns, UpSampling doubles the size
    x   = Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x   = UpSampling2D(2)(x)          # 16 → 32
    x   = Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x   = UpSampling2D(2)(x)          # 32 → 64
    x   = Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    x   = UpSampling2D(2)(x)          # 64 → 128
    #final layer: 3 colour channels, sigmoid keeps every pixel between 0 and 1
    decoded = Conv2DTranspose(3, 3, activation="sigmoid", padding="same",
                              name="decoded")(x)

    decoder = Model(dec_inp, decoded, name="decoder")

    # ── FULL AUTOENCODER, squeeze then redraw, glued together ──────────────
    ae_out = decoder(encoder(inp))
    autoencoder = Model(inp, ae_out, name="autoencoder")

    return autoencoder, encoder, decoder


def compile_autoencoder(autoencoder: "Model",
                        learning_rate: float = 1e-3) -> "Model":
    "set the knobs: Adam tutor + MSE punishment (predicting pixels = predicting numbers → MSE, not cross-entropy)"
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse"
    )
    return autoencoder


def reconstruction_error(autoencoder: "Model",
                          images: "np.ndarray") -> "np.ndarray":
    "the scoring bench: original minus redraw, squared, averaged → one error number per image. higher = more off-grade"
    #ask the intern to redraw every image
    reconstructed = autoencoder.predict(images, verbose=0)
    #average the squared pixel differences across height, width and colour → one number each
    mse = np.mean((images - reconstructed) ** 2, axis=(1, 2, 3))
    return mse
