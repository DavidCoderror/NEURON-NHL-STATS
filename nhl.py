import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load Excel
df = pd.read_excel("NHL_Season_Stats.xlsx", engine='openpyxl')

FEATURES = ['GP','W','L','OT','P','P%','S/O Win','SO','PP%','PK%','Shots/GP']

# Scale numeric features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES].values)

# Prepare sequences per team
seq_length = 3
X, y = [], []

teams = df['Team'].unique()

for team in teams:
    team_data = df_scaled[df_scaled['Team'] == team].sort_values('Season')[FEATURES].values


    if team_data.shape[0] != seq_length:
        print(f"Skipping {team}, unexpected number of seasons: {team_data.shape[0]}")
        continue

    X.append(team_data)        # shape = (3, num_features)
    y.append(team_data[-1])    # shape = (num_features,)


# Convert to arrays
X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(FEATURES))))
model.add(Dense(len(FEATURES)))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=100, verbose=0)

# Predict next season for a team
team_name = "Winnipeg Jets"
team_data = df_scaled[df_scaled['Team'] == team_name].sort_values('Season')[FEATURES].values

# Use last seq_length seasons
test_input = team_data[-seq_length:].reshape(1, seq_length, len(FEATURES))
scaled_pred = model.predict(test_input, verbose=0)[0]

# Convert back to original scale
pred = scaler.inverse_transform(scaled_pred.reshape(1, -1))[0]

# Print nicely
print(f"\nPredicted next season stats for {team_name}:\n")
for i, feat in enumerate(FEATURES):
    print(f"{feat}: {pred[i]:.2f}")
