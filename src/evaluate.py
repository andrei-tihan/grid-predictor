# src/evaluate.py

# Script to evaluate regression and classification model predictions.
# Provides functions to compute common metrics for regression and classification tasks.

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    dir_acc = (np.sign(y_true) == np.sign(y_pred)).mean()
    return {"mae": float(mae), "rmse": float(rmse), "dir_acc": float(dir_acc)}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "confusion_matrix": cm}