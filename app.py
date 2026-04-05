from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Allow frontend requests from different domain/port

#----------------------------------------------------------------------------------#

FEATURES = ['GP','W','L','OT','P','P%','S/O Win','SO','PP%','PK%','Shots/GP']
seq_length = 15

# Load data
df = pd.read_excel("nhl_stats_v2.xlsx", engine='openpyxl')

# Load models
lstm_model = load_model("NHL_LSTM.keras")
modelProb = joblib.load("NHL_Regr.pkl")
scaler_lr = joblib.load("NHL_Regr_scaler.pkl")

# Recreate scaler for LSTM 
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES].values)

#----------------------------------------------------------------------------------#

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    team_name = data.get('team')

    # Get team data
    team_data = df_scaled[df_scaled['Team'] == team_name].sort_values('Season')[FEATURES].values

    # Prepare input for LSTfM
    test_input = team_data[-seq_length:].reshape(1, seq_length, len(FEATURES))

    # Predict stats
    scaled_pred = lstm_model.predict(test_input, verbose=0)[0]
    pred = scaler.inverse_transform(scaled_pred.reshape(1, -1))[0]

    # Playoff probabilty
    pred_scaled_lr = scaler_lr.transform(pred.reshape(1, -1))
    prob = modelProb.predict_proba(pred_scaled_lr)[0][1]




if __name__ == '__main__': # IMPROTANT! FORGOT ABOUT THIS!
    app.run(debug=True)
