import numpy as np
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

# --------------------------------------------------
# Parameters
# --------------------------------------------------
NUM_WORDS = 10000
MAX_LEN = 250

# --------------------------------------------------
# Load IMDB dataset
# --------------------------------------------------
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=NUM_WORDS
)

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

# --------------------------------------------------
# Build model
# --------------------------------------------------
model = Sequential([
    Embedding(NUM_WORDS, 32, input_length=MAX_LEN),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model.summary()

# --------------------------------------------------
# Train model
# --------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=128
)

# --------------------------------------------------
# Save weights
# --------------------------------------------------
model.save_weights("sentiment_rnn.weights.h5")
