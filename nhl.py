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

# ------------------------------------------------------------------
# Enter the team name here Noah, look in excel to write proper name
team_name = "Pittsburgh Penguins"
# ------------------------------------------------------------------

team_data = df_scaled[df_scaled['Team'] == team_name].sort_values('Season')[FEATURES].values
test_input = team_data[-seq_length:].reshape(1, seq_length, len(FEATURES))

scaled_pred = model.predict(test_input, verbose=0)[0]

# Convert back
pred = scaler.inverse_transform(scaled_pred.reshape(1, -1))[0]

print(f"\nPredicted Season 4 stats for {team_name}:\n")
for i, feat in enumerate(FEATURES):
    print(f"{feat}: {pred[i]:.2f}")


#-----------------------------------------------------------------------------------
# Logistic Regression then takes the Test to maker a probabilty

df['Playoff'] = (df['P'] >= 80).astype(int)

from sklearn.linear_model import LogisticRegression

scaler_lr = MinMaxScaler()
X_lr_scaled = scaler_lr.fit_transform(df[FEATURES])
y_lr = df['Playoff']

modelProb = LogisticRegression(max_iter=1000, C=0.5)
modelProb.fit(X_lr_scaled, y_lr)

# Scale LSTM prediction to match training
pred_scaled_lr = scaler_lr.transform(pred.reshape(1, -1))
prob = modelProb.predict_proba(pred_scaled_lr)[0][1]

print(f"\nPlayoff probability for {team_name}: {prob*100:.1f}%")




#-----------------------------------------------------------------------------
# Save LSTM
from tensorflow.keras.models import load_model
model.save("NHL_LSTM")

# Save Regression
import joblib
joblib.dump(modelProb, "NHL_Regr.pkl")
joblib.dump(scaler_lr, "NHL_Regr_scaler.pkl")  # Must save the scaler too
