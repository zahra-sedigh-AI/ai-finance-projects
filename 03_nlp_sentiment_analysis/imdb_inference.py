import numpy as np
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

NUM_WORDS = 10000
MAX_LEN = 250

# Load word index
word_index = imdb.get_word_index()

# Build same model
model = Sequential([
    Embedding(NUM_WORDS, 32, input_length=MAX_LEN),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.load_weights("sentiment_rnn.weights.h5")

# --------------------------------------------------
# Convert sentence to sequence
# --------------------------------------------------
def encode_sentence(sentence):
    encoded = []
    for word in sentence.lower().split():
        encoded.append(word_index.get(word, 2))
    return pad_sequences([encoded], maxlen=MAX_LEN)

# --------------------------------------------------
# Test sentence
# --------------------------------------------------
sentence = "this movie was amazing and I really loved it"
sequence = encode_sentence(sentence)
prediction = model.predict(sequence)[0][0]

if prediction > 0.5:
    print("POSITIVE sentiment:", prediction)
else:
    print("NEGATIVE sentiment:", prediction)
