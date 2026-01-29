# LSTM-based Bitcoin Price Forecasting
# Academic Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data = pd.read_csv("data/btc_1d_data_2018_to_2025.csv")
data['Date'] = pd.to_datetime(data['Date'])

prices = data[['Close']].values

# --------------------------------------------------
# 2. Normalize data
# --------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# --------------------------------------------------
# 3. Create sequences
# --------------------------------------------------
def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices, lookback=10)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --------------------------------------------------
# 4. Train / Test split
# --------------------------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 5. Build LSTM model
# --------------------------------------------------
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --------------------------------------------------
# 6. Train model
# --------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# --------------------------------------------------
# 7. Plot training history
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("LSTM Training Loss")
plt.show()

# --------------------------------------------------
# 8. Predictions
# --------------------------------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)

y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# --------------------------------------------------
# 9. Plot predictions
# --------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label="Actual Price")
plt.plot(test_pred, label="Predicted Price")
plt.title("Bitcoin Price Prediction (Test Set)")
plt.legend()
plt.show()
