from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import Model

def create_model(vocab_size, max_length):
    # Image model
    inputs1 = Input(shape=(4096,))
    fe = Dropout(0.5)(inputs1)
    fe = Dense(256, activation="relu")(fe)

    # Text model
    inputs2 = Input(shape=(max_length,))
    se = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se = Dropout(0.5)(se)
    se = LSTM(256)(se)

    # Decoder
    decoder = add([fe, se])
    decoder = Dense(256, activation="relu")(decoder)
    outputs = Dense(vocab_size, activation="softmax")(decoder)

    model = Model([inputs1, inputs2], outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model
