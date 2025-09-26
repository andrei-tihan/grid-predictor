# src/features.py

# Script to create time-based and lag features for time series data.
# Assumes input dataframe has 'timestamp' and 'balance_mw' columns.
# Outputs dataframe with additional features.

import numpy as np

def make_features(df, lags=(1,24,168), rolls=(24,168)):
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
