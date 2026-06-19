"""
monitor_service.py
Runs a continuous monitoring loop every MONITOR_INTERVAL_SECONDS.
For each machine it:
  1. Generates a new sensor reading
  2. Scores it with the Isolation Forest
  3. Checks rule-based thresholds
  4. Persists the row and any alerts to SQLite + CSV
"""

import random
import logging
import signal
import sys
import time
from datetime import datetime

import config
import data_generator as dg
import model_trainer  as mt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/monitor.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Alert throttle: don't spam identical alerts within 60 s ──────────────────
_last_alert: dict[str, float] = {}
ALERT_COOLDOWN = 60  # seconds


def _can_alert(key: str) -> bool:
    now = time.time()
    if now - _last_alert.get(key, 0) >= ALERT_COOLDOWN:
        _last_alert[key] = now
        return True
    return False


def _check_thresholds(row: dict, anomaly_label: int, anomaly_score: float):
    mid  = row["machine_id"]
    name = row["machine_name"]
    ts   = row["timestamp"]

    alerts = []

    # Temperature
    if row["temperature"] >= config.THRESHOLDS["temperature_critical"]:
        alerts.append((mid, "High Temperature", "CRITICAL",
                       f"{name}: temperature {row['temperature']}°C exceeds critical limit "
                       f"({config.THRESHOLDS['temperature_critical']}°C)"))
    elif row["temperature"] >= config.THRESHOLDS["temperature_high"]:
        alerts.append((mid, "High Temperature", "WARNING",
                       f"{name}: temperature {row['temperature']}°C above threshold "
                       f"({config.THRESHOLDS['temperature_high']}°C)"))

    # Vibration
    if row["vibration"] >= config.THRESHOLDS["vibration_critical"]:
        alerts.append((mid, "Abnormal Vibration", "CRITICAL",
                       f"{name}: vibration {row['vibration']} mm/s exceeds critical limit "
                       f"({config.THRESHOLDS['vibration_critical']} mm/s)"))
    elif row["vibration"] >= config.THRESHOLDS["vibration_high"]:
        alerts.append((mid, "Abnormal Vibration", "WARNING",
                       f"{name}: vibration {row['vibration']} mm/s above threshold "
                       f"({config.THRESHOLDS['vibration_high']} mm/s)"))

    # Machine failure
    if row["status"] == "Failed":
        alerts.append((mid, "Machine Failure", "CRITICAL",
                       f"{name}: machine reported FAILED state"))

    # AI anomaly
    if anomaly_label == -1:
        alerts.append((mid, "AI Anomaly Detected", "WARNING",
                       f"{name}: anomaly score {anomaly_score:.4f} — unusual sensor pattern detected"))

    # Pressure
    if row["pressure"] >= config.THRESHOLDS["pressure_high"]:
        alerts.append((mid, "High Pressure", "WARNING",
                       f"{name}: pressure {row['pressure']} bar above limit "
                       f"({config.THRESHOLDS['pressure_high']} bar)"))

    # RPM
    if row["rpm"] >= config.THRESHOLDS["rpm_high"]:
        alerts.append((mid, "Overspeed", "WARNING",
                       f"{name}: RPM {row['rpm']} exceeds max ({config.THRESHOLDS['rpm_high']})"))
    elif row["rpm"] <= config.THRESHOLDS["rpm_low"] and row["status"] != "Failed":
        alerts.append((mid, "Low RPM / Stall", "WARNING",
                       f"{name}: RPM {row['rpm']} below minimum ({config.THRESHOLDS['rpm_low']})"))

    for mid_, atype, severity, msg in alerts:
        key = f"{mid_}:{atype}"
        if _can_alert(key):
            dg.insert_alert(mid_, atype, severity, msg)
            log.warning("[ALERT %s] %s — %s", severity, atype, msg)


def monitor_cycle(model, scaler):
    """One monitoring cycle: generate + score + persist readings for all machines."""
    now = datetime.now()
    log.info("── Monitor cycle at %s ──", now.strftime("%H:%M:%S"))

    for mid in config.MACHINES:
        # Inject occasional anomaly / failure for realism
        anomaly = random.random() < 0.04
        failure = random.random() < 0.008

        row = dg.generate_row(mid, now, anomaly=anomaly, failure=failure)

        label, score = mt.predict_anomaly(model, scaler, row)
        row["anomaly"] = 1 if (anomaly or failure or label == -1) else 0

        dg.append_log_row(row, anomaly_score=score)
        _check_thresholds(row, label, score)

        log.debug("  %s | temp=%.1f vib=%.2f status=%s score=%.4f",
                  mid, row["temperature"], row["vibration"], row["status"], score)


def run():
    """Block and run the monitor loop indefinitely."""
    import os
    os.makedirs("logs", exist_ok=True)

    log.info("Loading AI model …")
    model, scaler = mt.load_model()
    log.info("Monitor service started (interval: %ds)", config.MONITOR_INTERVAL_SECONDS)

    def _shutdown(sig, frame):
        log.info("Shutdown signal received — exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        try:
            monitor_cycle(model, scaler)
        except Exception as exc:
            log.error("Monitor cycle error: %s", exc, exc_info=True)
        time.sleep(config.MONITOR_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
