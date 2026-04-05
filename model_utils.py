import pickle
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ── Rebuild exact architecture ───────────────────────────────────────
def build_model():
    inputs = tf.keras.Input(shape=(123,), dtype="float32", name="input_layer")

    # Embedding with masking
    x = tf.keras.layers.Embedding(
        input_dim=31306,
        output_dim=50,
        mask_zero=True,
        name="embedding"
    )(inputs)

    # Compute mask manually (replaces old NotEqual layer)
    mask = tf.keras.layers.Lambda(
        lambda t: tf.cast(tf.not_equal(t, 0), dtype=tf.bool),
        name="not_equal"
    )(inputs)

    # Bidirectional LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=128,
            return_sequences=True,
            zero_output_for_mask=True,
            name="forward_lstm"
        ),
        merge_mode="concat",
        name="bidirectional"
    )(x, mask=mask)

    # POS output — 35 classes
    pos_out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(35, activation="softmax", name="dense"),
        name="pos"
    )(x)

    # NER output — 12 classes
    ner_out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(12, activation="softmax", name="dense_1"),
        name="ner"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=[pos_out, ner_out])
    return model


# ── Load weights using exact verified h5 paths ───────────────────────
def load_weights_from_h5(model, h5_path):
    with h5py.File(h5_path, "r") as f:
        mw = f["model_weights"]

        def get_w(path):
            return np.array(mw[path])

        # ── Embedding ────────────────────────────────────────────────
        model.get_layer("embedding").set_weights([
            get_w("embedding/embedding/embeddings")
        ])

        # ── Bidirectional LSTM ───────────────────────────────────────
        fwd_kernel = get_w("bidirectional/bidirectional/forward_lstm/lstm_cell/kernel")
        fwd_rec    = get_w("bidirectional/bidirectional/forward_lstm/lstm_cell/recurrent_kernel")
        fwd_bias   = get_w("bidirectional/bidirectional/forward_lstm/lstm_cell/bias")
        bwd_kernel = get_w("bidirectional/bidirectional/backward_lstm/lstm_cell/kernel")
        bwd_rec    = get_w("bidirectional/bidirectional/backward_lstm/lstm_cell/recurrent_kernel")
        bwd_bias   = get_w("bidirectional/bidirectional/backward_lstm/lstm_cell/bias")

        model.get_layer("bidirectional").set_weights([
            fwd_kernel, fwd_rec, fwd_bias,
            bwd_kernel, bwd_rec, bwd_bias
        ])

        # ── POS Dense ────────────────────────────────────────────────
        model.get_layer("pos").set_weights([
            get_w("pos/pos/dense/kernel"),
            get_w("pos/pos/dense/bias")
        ])

        # ── NER Dense ────────────────────────────────────────────────
        model.get_layer("ner").set_weights([
            get_w("ner/ner/dense_1/kernel"),
            get_w("ner/ner/dense_1/bias")
        ])

    print("✅ All weights loaded successfully")
    return model


# ── Load all artifacts ───────────────────────────────────────────────
def load_artifacts():
    model = build_model()
    model = load_weights_from_h5(model, "pos_ner_model.h5")

    with open("tokenizer_sen.pkl", "rb") as f:
        tok_sen = pickle.load(f)
    with open("tokenizer_pos.pkl", "rb") as f:
        tok_pos = pickle.load(f)
    with open("tokenizer_ner.pkl", "rb") as f:
        tok_ner = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tok_sen, tok_pos, tok_ner, max_len


# ── Prediction ───────────────────────────────────────────────────────
def predict(sentence: str, model, tok_sen, tok_pos, tok_ner, max_len):
    tokens = sentence.strip().split()

    seq    = tok_sen.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=max_len, padding="post")

    pos_pred, ner_pred = model.predict(padded, verbose=0)

    pos_idx = np.argmax(pos_pred[0], axis=-1)[:len(tokens)]
    ner_idx = np.argmax(ner_pred[0], axis=-1)[:len(tokens)]

    pos_inv = {v: k for k, v in tok_pos.word_index.items()}
    ner_inv = {v: k for k, v in tok_ner.word_index.items()}

    results = []
    for word, p, n in zip(tokens, pos_idx, ner_idx):
        results.append({
            "Token": word,
            "POS":   pos_inv.get(int(p), "UNK"),
            "NER":   ner_inv.get(int(n), "O"),
        })
    return results