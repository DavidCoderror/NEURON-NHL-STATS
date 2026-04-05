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
lstm_model = load_model("NHL_LSTM")
modelProb = joblib.load("NHL_Regr.pkl")
scaler_lr = joblib.load("NHL_Regr_scaler.pkl")
