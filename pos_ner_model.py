import numpy as np
import pandas as pd
import pickle
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional)
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0)) 

# ── 1. Load & parse ───────────────────────────────────────────────────────────
df = pd.read_csv("NER_dataset.csv", encoding='latin1')

Sentences, PartOfSpeech, NER = [], [], []
s, p, n = "", "", ""

for _, row in df.iterrows():
    temp = row['Sentence #']
    if isinstance(temp, str) and temp != 'Sentence: 1':
        if s.strip():
            Sentences.append(s.strip())
            PartOfSpeech.append(p.strip())
            NER.append(n.strip())
        s, p, n = "", "", ""
    if isinstance(row['Word'], str) and row['Word'] != '.':
        s += row['Word'] + " "
        p += row['POS'] + " "
        n += row['Tag'] + " "

# Append last sentence
if s.strip():
    Sentences.append(s.strip())
    PartOfSpeech.append(p.strip())
    NER.append(n.strip())

# ── 2. Clean sentences only (NOT POS/NER tags) ────────────────────────────────
translator = str.maketrans('', '', string.punctuation)
Sentences = [' '.join(s.lower().translate(translator).split()) for s in Sentences]

# ── 3. Tokenize ───────────────────────────────────────────────────────────────
tok_sen = Tokenizer()
tok_sen.fit_on_texts(Sentences)

tok_pos = Tokenizer()
tok_pos.fit_on_texts(PartOfSpeech)

tok_ner = Tokenizer()
tok_ner.fit_on_texts(NER)

tok_sentence = tok_sen.texts_to_sequences(Sentences)
tok_pos_seq  = tok_pos.texts_to_sequences(PartOfSpeech)
tok_ner_seq  = tok_ner.texts_to_sequences(NER)

# ── 4. Fix: single consistent max_len ────────────────────────────────────────
max_len = max(
    max(len(i) for i in tok_sentence),
    max(len(i) for i in tok_pos_seq),
    max(len(i) for i in tok_ner_seq)
)

X_padded     = pad_sequences(tok_sentence, maxlen=max_len, padding='post')
Y_pos_padded = pad_sequences(tok_pos_seq,  maxlen=max_len, padding='post')
Y_ner_padded = pad_sequences(tok_ner_seq,  maxlen=max_len, padding='post')

Y_pos_padded = np.expand_dims(Y_pos_padded, -1)
Y_ner_padded = np.expand_dims(Y_ner_padded, -1)

# ── 5. Multi-output model for joint POS + NER ─────────────────────────────────
vocab_size = len(tok_sen.word_index) + 1
num_pos    = len(tok_pos.word_index) + 1
num_ner    = len(tok_ner.word_index) + 1
emb_dim    = 50
lstm_units = 128

inputs    = Input(shape=(max_len,))
embedding = Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len, mask_zero=True)(inputs)
bilstm    = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding)

pos_out = TimeDistributed(Dense(num_pos, activation='softmax'), name='pos')(bilstm)
ner_out = TimeDistributed(Dense(num_ner, activation='softmax'), name='ner')(bilstm)

model = Model(inputs=inputs, outputs=[pos_out, ner_out])
model.compile(
    optimizer='adam',
    loss={'pos': 'sparse_categorical_crossentropy',
          'ner': 'sparse_categorical_crossentropy'},
    metrics={'pos': ['accuracy'], 'ner': ['accuracy']}
)

model.summary()

# ── 6. Train ──────────────────────────────────────────────────────────────────
history = model.fit(
    X_padded,
    {'pos': Y_pos_padded, 'ner': Y_ner_padded},
    epochs=30,          
    batch_size=128,
    validation_split=0.2
)

print("Final POS Accuracy:", history.history['pos_accuracy'][-1])
print("Final NER Accuracy:", history.history['ner_accuracy'][-1])

# ── 7. Save ───────────────────────────────────────────────────────────────────
model.save("pos_ner_model.h5")
for name, tok in [("tokenizer_sen", tok_sen), ("tokenizer_pos", tok_pos),
                  ("tokenizer_ner", tok_ner)]:
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(tok, f)
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

index_to_pos = {v: k for k, v in tok_pos.word_index.items()}
index_to_ner = {v: k for k, v in tok_ner.word_index.items()}

def predictor(sentence):
    sentence = sentence.lower().translate(translator)
    seq = tok_sen.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    pred_pos, pred_ner = model.predict(padded_seq)
    pred_pos_labels = [index_to_pos.get(np.argmax(pos), 'O') for pos in pred_pos[0]]
    pred_ner_labels = [index_to_ner.get(np.argmax(ner), 'O') for ner in pred_ner[0]]
    return list(zip(sentence.split(), pred_pos_labels, pred_ner_labels))

print(predictor("Apple is looking at buying U.K. startup for $1 billion"))