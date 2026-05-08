import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load & preprocess data
df = pd.read_csv("HousingData.csv").rename(columns={"MEDV": "PRICE"})
df.fillna(df.mean(numeric_only=True), inplace=True)

# Visualizations
sns.displot(df["PRICE"])
plt.show()
# plt.figure(figsize=(15, 13))
# sns.heatmap(df.corr(), annot=True, square=True)
# plt.show()

# Features & target
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)
# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Training
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.05,  # Use 5% of the training data for validation to monitor overfitting
    verbose=1
)

# Prediction
y_pred = model.predict(X_test)
# Plot results
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.show()

# Evaluating the model
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

r2 = r2_score(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score =", r2_score(y_test, y_pred))

print("MSE =", mean_squared_error(y_test, y_pred))

print("RMSE =", rmse)

print("MAE =", mean_absolute_error(y_test, y_pred))
