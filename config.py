# config.py — Central configuration for AI Maintenance System

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

CSV_PATH   = os.path.join(DATA_DIR, "machine_logs.csv")
DB_PATH    = os.path.join(DATA_DIR, "maintenance.db")
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
SCALER_PATH= os.path.join(MODEL_DIR, "scaler.joblib")

# ── Machine definitions ────────────────────────────────────────────────────────
MACHINES = {
    "M001": {"name": "CNC Milling Machine",      "type": "CNC",        "location": "Bay A"},
    "M002": {"name": "Hydraulic Press",           "type": "Press",      "location": "Bay A"},
    "M003": {"name": "Conveyor Belt System",      "type": "Conveyor",   "location": "Bay B"},
    "M004": {"name": "Industrial Robot Arm",      "type": "Robot",      "location": "Bay B"},
    "M005": {"name": "Air Compressor",            "type": "Compressor", "location": "Bay C"},
    "M006": {"name": "Laser Cutting Machine",     "type": "Laser",      "location": "Bay C"},
    "M007": {"name": "Injection Moulding Machine","type": "Moulding",   "location": "Bay D"},
    "M008": {"name": "Electric Motor Drive",      "type": "Motor",      "location": "Bay D"},
    "M009": {"name": "Cooling Tower Unit",        "type": "Cooling",    "location": "Bay E"},
    "M010": {"name": "Welding Robot",             "type": "Welding",    "location": "Bay E"},
}

# ── Sensor thresholds ──────────────────────────────────────────────────────────
THRESHOLDS = {
    "temperature_high":   85.0,   # °C
    "temperature_critical":95.0,  # °C
    "vibration_high":      7.5,   # mm/s
    "vibration_critical": 12.0,   # mm/s
    "pressure_high":       8.5,   # bar
    "current_high":       42.0,   # A
    "rpm_low":           400.0,   # RPM — stall warning
    "rpm_high":         3200.0,   # RPM — overspeed
}

# ── Simulation parameters ──────────────────────────────────────────────────────
MONITOR_INTERVAL_SECONDS = 5
LOG_ROWS_PER_BATCH        = 1        # rows added per monitor cycle per machine
SEED                      = 42
HISTORY_HOURS             = 24       # hours of initial seed data

# ── Model parameters ──────────────────────────────────────────────────────────
CONTAMINATION = 0.05   # expected anomaly fraction for Isolation Forest
N_ESTIMATORS  = 200
