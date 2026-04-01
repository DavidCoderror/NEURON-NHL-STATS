import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example fake data (replace with your real dataset)
data = np.array([
    [82, 45, 25, 12, 102, 0.622, 5, 10, 20.5, 82.1, 31.2],
    [82, 50, 20, 12, 112, 0.683, 6, 8, 22.1, 83.5, 32.0],
    [82, 42, 30, 10, 94, 0.573, 4, 9, 19.8, 80.2, 30.5],
    [82, 55, 18, 9, 119, 0.726, 7, 7, 24.3, 85.0, 33.1],
    [82, 53, 20, 9, 115, 0.701, 6, 6, 23.5, 84.2, 32.8],
    [82, 58, 15, 9, 125, 0.762, 8, 5, 25.0, 86.1, 34.0],
])

FEATURES = ['GP','W','L','OT','P','P%','S/O Win','SO','PP%','PK%','Shots/GP']

# Prepare sequences
X, y = [], []
seq_length = 3

for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Build model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 11)))
model.add(Dense(11))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=50, verbose=0)

# Predict next season
test_input = data[-3:].reshape((1, seq_length, 11))
prediction = model.predict(test_input)[0]  # remove batch dimension

print("\nPredicted Next Season Stats:\n")
for i, feature in enumerate(FEATURES):
    print(f"{feature}: {prediction[i]:.2f}")
