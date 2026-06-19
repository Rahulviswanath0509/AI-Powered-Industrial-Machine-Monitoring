"""
model_trainer.py
Trains (or reloads) an Isolation Forest for anomaly detection
on the sensor features: temperature, vibration, pressure, current, rpm.
"""

import os
import logging
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURES = ["temperature", "vibration", "pressure", "current", "rpm"]


def _load_training_data() -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_PATH)
    df   = pd.read_sql("SELECT * FROM machine_logs", conn)
    conn.close()
    return df


def train_model(force: bool = False):
    """
    Train an Isolation Forest on all available historical data.
    Saves model + scaler to MODEL_DIR.
    Returns (model, scaler).
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if not force and os.path.exists(config.MODEL_PATH) and os.path.exists(config.SCALER_PATH):
        log.info("Loading existing model from %s", config.MODEL_PATH)
        model  = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        return model, scaler

    log.info("Training Isolation Forest …")
    df = _load_training_data()

    if df.empty:
        raise RuntimeError("No training data found. Run data_generator first.")

    X = df[FEATURES].dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=config.N_ESTIMATORS,
        contamination=config.CONTAMINATION,
        random_state=config.SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    joblib.dump(model,  config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    log.info("Model saved to %s", config.MODEL_PATH)
    return model, scaler


def load_model():
    """Load pre-trained model and scaler; train if missing."""
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.SCALER_PATH)):
        log.warning("Model not found — training now.")
        return train_model()
    model  = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler


def predict_anomaly(model, scaler, row: dict) -> tuple[int, float]:
    """
    Score a single sensor reading.
    Returns (label, raw_score):
        label  -1 = anomaly, 1 = normal
        score  lower → more anomalous (IsolationForest convention)
    """
    X = np.array([[row[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)
    label = model.predict(X_scaled)[0]          # -1 or 1
    score = model.score_samples(X_scaled)[0]    # more negative ⟹ more anomalous
    return int(label), float(score)


def batch_predict(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """Add 'pred_label' and 'anomaly_score' columns to a DataFrame."""
    X = df[FEATURES].fillna(0).values
    X_scaled = scaler.transform(X)
    df = df.copy()
    df["pred_label"]   = model.predict(X_scaled)
    df["anomaly_score"] = model.score_samples(X_scaled)
    return df


if __name__ == "__main__":
    model, scaler = train_model(force=True)
    print("Training complete.")
