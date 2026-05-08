import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and visualize data
df = pd.read_csv('goog1.csv')
close = df[['Close']].values

plt.plot(close)
plt.title("Closing Prices")
plt.show()

# Scale data
scaled = MinMaxScaler().fit_transform(close)

# Train-test split
split = int(len(scaled) * 0.8)
train, test = scaled[:split], scaled[split:]

# Create sequences
def create_dataset(data, step=4):
    X, y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i+step])
        y.append(data[i+step])
    return np.array(X), np.array(y)

x_train, y_train = create_dataset(train)
x_test, y_test = create_dataset(test)

# Build model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train and evaluate
model.fit(x_train, y_train, epochs=50, verbose=1)
print("Test Loss:", model.evaluate(x_test, y_test))

# Predict and inverse scale
scaler = MinMaxScaler().fit(close)
y_pred = scaler.inverse_transform(model.predict(x_test))
y_test = scaler.inverse_transform(y_test)

# Plot results
plt.plot(y_pred, label="Predicted")
plt.plot(y_test, label="Actual")
plt.legend()
plt.show()
