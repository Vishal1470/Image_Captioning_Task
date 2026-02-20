import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("image_caption_model.h5")

with open("image_features.pkl", "rb") as f:
    features = pickle.load(f)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34

def idx_to_word(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(photo):
    text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = np.argmax(model.predict([photo, seq], verbose=0))
        word = idx_to_word(yhat)
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text

image_id = list(features.keys())[0]
print("Generated Caption:")
print(generate_caption(features[image_id]))
