import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load Excel
df = pd.read_excel("nhl_stats_v2.xlsx", engine='openpyxl')

FEATURES = ['GP','W','L','OT','P','P%','S/O Win','SO','PP%','PK%','Shots/GP']

# Scale numeric features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES].values)

# Prepare sequences per team
seq_length = 15
X, y = [], []

teams = df['Team'].unique()

for team in teams:
    team_data = df_scaled[df_scaled['Team'] == team].sort_values('Season')[FEATURES].values

    for i in range(len(team_data) - seq_length):
        X.append(team_data[i:i+seq_length])
        y.append(team_data[i+seq_length])


# Convert to arrays
X = np.array(X)
y = np.array(y)
print("X shape:", X.shape) # Debug
print("y shape:", y.shape) # Debug

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, len(FEATURES))))
model.add(Dense(len(FEATURES)))
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=100, verbose=0)

#-----------------------------------------------------------------------------------
# Testing Zone For LSTM

team_name = "Ottawa Senators"

team_data = df_scaled[df_scaled['Team'] == team_name].sort_values('Season')[FEATURES].values

test_input = team_data[-seq_length:].reshape(1, seq_length, len(FEATURES))

scaled_pred = model.predict(test_input, verbose=0)[0]

# Convert back
pred = scaler.inverse_transform(scaled_pred.reshape(1, -1))[0]

print(f"\nPredicted Season 4 stats for {team_name}:\n")
for i, feat in enumerate(FEATURES):
    print(f"{feat}: {pred[i]:.2f}")


#-----------------------------------------------------------------------------------
# Logistic Regression

df['Playoff'] = (df['P'] >= 80).astype(int)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000,C=0.5)
clf.fit(df[FEATURES], df['Playoff'])

pred_df = pd.DataFrame([pred], columns=FEATURES)
prob = clf.predict_proba(pred_df)[0][1]

print(f"\nPlayoff probability for {team_name}: {prob*100:.1f}%")
