"""Algorithm 1 / Fig. 4: the 1D CNN (layers L1..L8), its initialisation, compilation and training."""
import numpy as np


def build_cnn(input_len, n_classes, cfg):
    import keras
    from keras import layers

    input_layer = keras.Input(shape=(input_len, 1))                                            # L1
    conv = layers.Conv1D(filters=cfg.conv_filters, kernel_size=cfg.kernel_size,
                         activation="relu")(input_layer)                                       # L2
    pool = layers.MaxPooling1D(pool_size=cfg.pool_size)(conv)                                  # L3
    flat = layers.Flatten()(pool)                                                              # L4
    dense = layers.Dense(cfg.dense_units, activation="relu")(flat)                             # L5
    if cfg.architecture == "fig4":
        # Fig. 4 and Section 3.2 text: Dense -> Dropout -> BatchNormalization -> Output.
        hidden = layers.Dropout(cfg.dropout)(dense)                                            # L6
        hidden = layers.BatchNormalization()(hidden)                                           # L7
    elif cfg.architecture == "algorithm1_literal":
        # Algorithm 1 as printed: L6 and L7 are created but never called, and line 8 feeds
        # dense_layer_1 straight into the output layer, so they do not take part in the model.
        layers.Dropout(cfg.dropout)                                                            # L6
        layers.BatchNormalization()                                                            # L7
        hidden = dense
    else:
        raise ValueError(f"unknown architecture {cfg.architecture!r}")
    output_layer = layers.Dense(n_classes, activation="softmax")(hidden)                       # L8
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
                  metrics=["accuracy"])
    return model


def as_cnn_input(X):
    return np.asarray(X, np.float32)[..., None]


def one_hot(y, n_classes):
    return np.eye(n_classes, dtype=np.float32)[np.asarray(y, np.int64)]


def train_cnn(model, X, y, n_classes, cfg):
    """Algorithm 1 line 11: model.fit(x_train, y_train, batch_size=512, epochs=10).

    `use_multiprocessing=True` from the paper only affected Keras 2 generator inputs; Keras 3 removed the
    argument for array inputs, so it is not passed (no effect on training with in-memory arrays).
    """
    model.fit(as_cnn_input(X), one_hot(y, n_classes), batch_size=cfg.batch_size, epochs=cfg.epochs,
              verbose=cfg.fit_verbose, shuffle=True)
    return model


def predict_classes(model, X, batch_size=4096):
    return np.argmax(model.predict(as_cnn_input(X), batch_size=batch_size, verbose=0), axis=1)


def accuracy_with_weights(eval_model, weights, X, y):
    """Step 8: the CNN Evaluation Model is pre-initialised with the given weights and evaluated."""
    eval_model.set_weights([np.asarray(w, np.float32) for w in weights])
    probs = eval_model.predict(as_cnn_input(X), batch_size=4096, verbose=0)
    if not np.all(np.isfinite(probs)):
        return float("nan")                     # e.g. a negative BatchNorm variance after noise
    return float(np.mean(np.argmax(probs, axis=1) == np.asarray(y)))
