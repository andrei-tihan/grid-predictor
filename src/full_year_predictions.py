# src/generate_full_year_predictions.py

# Script to generate full-year predictions using the final ANN model.


"""
Generate predictions for all timestamps in the dataset using the trained ANN model.
Outputs a CSV: data/processed/predictions_full_year.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

def make_features(df, lags=(1,24,168), rolls=(24,168)):
    df['timestamp'] = pd.to_datetime(df[date_col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True).copy()
    ts = df['timestamp']
    df['hour'] = ts.dt.hour
    df['dow'] = ts.dt.dayofweek
    df['month'] = ts.dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['day_of_year'] = ts.dt.dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    for lag in lags:
        df[f'balance_lag{lag}'] = df['balance_mw'].shift(lag)
    for win in rolls:
        df[f'balance_roll_{win}'] = df['balance_mw'].rolling(win).mean().shift(1)

    df['balance_delta1'] = df['balance_mw'] - df['balance_lag1']

    df = df.dropna().reset_index(drop=True)
    return df

def feature_list(df):
    exclude = {'timestamp','total_generation_mw','load_mw','balance_mw','wind_mw','solar_mw','other_gen_mw','label'}
    feats = [c for c in df.columns if c not in exclude and df[c].dtype.kind in 'biufc']
    return feats

# Paths
CLEAN_PATH = Path("data/processed/alberta_hourly_clean.csv")
MODEL_PATH = Path("models/ann_model.joblib")  # Replace with your ANN model filename
OUTPUT_PATH = Path("data/processed/predictions_full_year.csv")

# Load cleaned hourly data
df = pd.read_csv(CLEAN_PATH, parse_dates=["timestamp"])

# Optional: sort by timestamp to be safe
df = df.sort_values("timestamp").reset_index(drop=True)

# Feature engineering (reuse your training pipeline function)
X_full = make_features(df)

# Load trained ANN model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained ANN model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Generate predictions
y_pred = model.predict(X_full)

# Add predictions to dataframe
df['y_pred'] = y_pred

# Optionally, create prediction bands (like in your current code)
# Here we just use a simple placeholder +/- some error estimate
df['y_lower'] = df['y_pred'] - 10  # replace with actual uncertainty if available
df['y_upper'] = df['y_pred'] + 10

# Optional: create bands for simple classification
def assign_band(val):
    if val < -50:
        return "Deficit"
    elif val > 50:
        return "Surplus"
    else:
        return "Neutral"

df['band'] = df['y_pred'].apply(assign_band)

# Placeholder recommendations (you can expand these later)
def grid_action(band):
    return {
        "Deficit": "Reduce load or dispatch storage",
        "Neutral": "Maintain operations",
        "Surplus": "Charge storage or incentivize demand"
    }.get(band, "N/A")

def customer_action(band):
    return {
        "Deficit": "Delay EV charging / smart appliance use",
        "Neutral": "Normal operation",
        "Surplus": "Charge EVs or use flexible appliances"
    }.get(band, "N/A")

df['grid_recommendation'] = df['band'].apply(grid_action)
df['customer_recommendation'] = df['band'].apply(customer_action)

# Save to CSV
df.to_csv(OUTPUT_PATH, index=False)
print(f"Full-year predictions saved to {OUTPUT_PATH}")