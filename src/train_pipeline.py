# src/train_pipeline.py
"""
Script to run the full training pipeline:

Usage (from repo root):
python -m src.train_pipeline --gen data/raw/alberta_net_generation.csv --load data/raw/alberta_internal_load.csv

Outputs:
 - data/processed/alberta_hourly_clean.csv
 - data/processed/predictions.csv
 - models/* (saved models)
 - models/meta.json
"""

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from . import data_prep, features, recommend, evaluate, utils
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


def build_ann_regressor_sklearn():
    from sklearn.neural_network import MLPRegressor
    return ("sklearn_mlp", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))

def build_ann_regressor_keras(input_dim):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adamW', loss='mae')
    return ("keras", model)

def main(args):
    cfg = utils.load_yaml(args.config) if args.config else {
        "tz":"America/Edmonton","test_days":90,"lags":[1,24,168],"rolls":[24,168],
        "neutral_k":0.25,"high_k":0.75,"models":{"regressor":"hgb","classifier":"rf"},"n_clusters_daily":3
    }

    # Directories
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Data loading and preparation (produces data/processed/alberta_hourly_clean.csv)
    print("Loading and cleaning data...")
    df_clean = data_prep.load_and_prepare(args.gen, args.load, Path("data/processed/alberta_hourly_clean.csv"))

    # Features
    df_feat = features.make_features(df_clean, lags=tuple(cfg.get("lags",[1,24,168])), rolls=tuple(cfg.get("rolls",[24,168])))
    feats = features.feature_list(df_feat)
    target = 'balance_mw'

    # Train/test split (time-based)
    last_ts = df_feat['timestamp'].max()
    cutoff = last_ts - pd.Timedelta(days=cfg.get("test_days",90))
    train_df = df_feat[df_feat['timestamp'] <= cutoff].reset_index(drop=True)
    test_df  = df_feat[df_feat['timestamp'] > cutoff].reset_index(drop=True)
    print("Train rows:", len(train_df), "Test rows:", len(test_df))

    Xtr = train_df[feats]
    ytr = train_df[target]
    Xte = test_df[feats]
    yte = test_df[target]

    sigma_train = float(ytr.std())

    # Regression model selection
    reg_choice = cfg.get("models",{}).get("regressor","ann")
    print("Training regressor:", reg_choice)
    if reg_choice == "hgb":
        reg = HistGradientBoostingRegressor(random_state=42)
        reg.fit(Xtr, ytr)
        ypred = reg.predict(Xte)
        joblib.dump(reg, "models/regressor.joblib")
    elif reg_choice == "ridge":
        reg = Ridge(random_state=42)
        reg.fit(Xtr, ytr)
        ypred = reg.predict(Xte)
        joblib.dump(reg, "models/regressor.joblib")
    elif reg_choice == "ann":
        backend, model = build_ann_regressor_keras(Xtr.shape[1])
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        early_stop = EarlyStopping(
            monitor='val_loss',  # Which metric to monitor
            patience=10,  # Stop after 10 epochs with no improvement
            restore_best_weights=True
        )
        history = model.fit(Xtr_s, ytr, validation_split=0.1, epochs=100, batch_size=32, verbose=2, callbacks=[early_stop])
        with open('models/ann_history.json', 'w') as f:
            json.dump(history.history, f)
        ypred = model.predict(Xte_s).ravel()
        model.save("models/regressor.keras")
        joblib.dump(scaler, "models/scaler_ann.joblib")
    else:
        # default to hgb
        reg = HistGradientBoostingRegressor(random_state=42)
        reg.fit(Xtr, ytr)
        ypred = reg.predict(Xte)
        joblib.dump(reg, "models/regressor.joblib")

    reg_metrics = evaluate.regression_metrics(yte, ypred)
    print("Regression metrics:", reg_metrics)

    # compute resid std on training (for band)
    try:
        if reg_choice == "ann":
            scaler = joblib.load("models/scaler_ann.joblib")
            train_pred = model.predict(scaler.transform(Xtr)).ravel()
        else:
            train_pred = reg.predict(Xtr)
        resid_std = float((ytr - train_pred).std())
    except Exception:
        resid_std = float((ytr - reg.predict(Xtr)).std())

    # Step 5: classification (3-class)
    neutral_k = cfg.get("neutral_k", 0.25)
    high_k = cfg.get("high_k", 0.75)
    def label_from_balance(x):
        if x >= high_k * sigma_train: return 2
        if x <= -high_k * sigma_train: return 0
        if abs(x) <= neutral_k * sigma_train: return 1
        return 2 if x > 0 else 0

    df_feat['label'] = df_feat['balance_mw'].apply(label_from_balance)
    train_df = df_feat[df_feat['timestamp'] <= cutoff].reset_index(drop=True)
    test_df  = df_feat[df_feat['timestamp'] > cutoff].reset_index(drop=True)
    clf_feats = feats
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(train_df[clf_feats], train_df['label'])
    ypred_clf = clf.predict(test_df[clf_feats])
    clf_metrics = evaluate.classification_metrics(test_df['label'], ypred_clf)
    print("Classifier metrics:", clf_metrics)
    joblib.dump(clf, "models/classifier.joblib")

    # Step 6: clustering daily profiles
    daily = df_clean[['timestamp','balance_mw']].copy()
    daily['date'] = daily['timestamp'].dt.date
    daily['hour'] = daily['timestamp'].dt.hour
    daily_pivot = daily.pivot_table(
        index='date',
        columns='hour',
        values='balance_mw',
        aggfunc='mean'
    ).fillna(method='ffill', axis=1)
    sc = StandardScaler()
    X_daily = sc.fit_transform(daily_pivot.fillna(0))
    n_clusters = cfg.get("n_clusters_daily", 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    daily_labels = kmeans.fit_predict(X_daily)
    sil = silhouette_score(X_daily, daily_labels)
    joblib.dump(kmeans, "models/kmeans_daily.joblib")

    # Build predictions.csv for app
    out = test_df[['timestamp','balance_mw']].copy()
    out['y_pred'] = ypred
    out['y_lower'] = out['y_pred'] - resid_std
    out['y_upper'] = out['y_pred'] + resid_std
    out['band'] = out['y_pred'].apply(lambda x: recommend.band_label(x, sigma_train, neutral_k=neutral_k, high_k=high_k))
    out['grid_description'] = out['band'].apply(lambda b: recommend.recommendation_from_band(b)['description'])
    out['grid_recommendation'] = out['band'].apply(lambda b: recommend.recommendation_from_band(b)['grid_action'])
    out['customer_recommendation'] = out['band'].apply(lambda b: recommend.recommendation_from_band(b)['customer_action'])
    out.to_csv("data/processed/predictions.csv", index=False)
    print("Wrote predictions.csv (for Streamlit app)")

    # Step 8: save meta
    meta = {
        "regression_metrics": reg_metrics,
        "classification_metrics": clf_metrics,
        "clustering_silhouette": float(sil),
        "sigma_train": float(sigma_train),
        "resid_std": float(resid_std),
        "features": feats
    }
    utils.save_json("models/meta.json", meta)
    print("Saved models/meta.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", required=True, help="path to generation CSV")
    parser.add_argument("--load", required=True, help="path to load CSV")
    parser.add_argument("--config", default="config.yaml", help="path to config yaml")
    args = parser.parse_args()
    main(args)
