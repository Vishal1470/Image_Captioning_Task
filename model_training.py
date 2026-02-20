import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

with open("cleaned_captions.pkl", "rb") as f:
    captions = pickle.load(f)

with open("image_features.pkl", "rb") as f:
    features = pickle.load(f)

all_captions = []
for caps in captions.values():
    all_captions.extend(caps)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

def create_sequences(caps, photo):
    X1, X2, y = [], [], []
    for cap in caps:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
            out_seq = to_categorical(seq[i], vocab_size)
            X1.append(photo[0])
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

inputs1 = Input(shape=(4096,))
fe = Dropout(0.5)(inputs1)
fe = Dense(256, activation="relu")(fe)

inputs2 = Input(shape=(max_length,))
se = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se = Dropout(0.5)(se)
se = LSTM(256)(se)

decoder = add([fe, se])
decoder = Dense(256, activation="relu")(decoder)
outputs = Dense(vocab_size, activation="softmax")(decoder)

model = Model([inputs1, inputs2], outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam")

for key in list(captions.keys())[:100]:  # limit for testing
    X1, X2, y = create_sequences(captions[key], features[key])
    model.fit([X1, X2], y, epochs=1, verbose=1)

model.save("image_caption_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model trained and saved")
