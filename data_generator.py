"""
data_generator.py
Generates realistic sensor logs for 10 industrial machines,
seeds historical data, and initialises the SQLite schema.
"""

import os
import sqlite3
import random
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Per-machine "normal" operating envelopes ──────────────────────────────────
MACHINE_PROFILES = {
    "M001": dict(temp=(55, 8),  vib=(2.5, 0.6), pressure=(4.5, 0.4), current=(28, 3), rpm=(1800, 120)),
    "M002": dict(temp=(60, 7),  vib=(3.0, 0.7), pressure=(6.0, 0.5), current=(35, 4), rpm=(900,  80)),
    "M003": dict(temp=(45, 5),  vib=(1.8, 0.4), pressure=(2.5, 0.3), current=(18, 2), rpm=(600,  50)),
    "M004": dict(temp=(50, 6),  vib=(2.2, 0.5), pressure=(3.5, 0.4), current=(22, 3), rpm=(1200, 100)),
    "M005": dict(temp=(70, 9),  vib=(4.0, 0.8), pressure=(7.5, 0.6), current=(38, 4), rpm=(2800, 150)),
    "M006": dict(temp=(65, 8),  vib=(1.5, 0.3), pressure=(5.0, 0.4), current=(30, 3), rpm=(3000, 100)),
    "M007": dict(temp=(75, 10), vib=(3.5, 0.7), pressure=(6.5, 0.5), current=(40, 4), rpm=(1500, 120)),
    "M008": dict(temp=(55, 7),  vib=(2.8, 0.6), pressure=(4.0, 0.4), current=(25, 3), rpm=(2200, 130)),
    "M009": dict(temp=(35, 5),  vib=(1.2, 0.3), pressure=(2.0, 0.2), current=(15, 2), rpm=(800,  60)),
    "M010": dict(temp=(80, 10), vib=(5.0, 1.0), pressure=(5.5, 0.5), current=(42, 4), rpm=(1000, 90)),
}

STATUS_OPTIONS = ["Running", "Running", "Running", "Running", "Warning", "Critical", "Failed"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def generate_row(machine_id: str, timestamp: datetime,
                 anomaly: bool = False, failure: bool = False) -> dict:
    """Return one sensor reading row for a machine."""
    p = MACHINE_PROFILES[machine_id]
    rng = random.Random()

    def sample(mu_sig, lo, hi):
        v = rng.gauss(mu_sig[0], mu_sig[1])
        if anomaly:
            v += rng.uniform(mu_sig[1] * 3, mu_sig[1] * 6)
        return round(_clamp(v, lo, hi), 2)

    temp     = sample(p["temp"],     5,   120)
    vib      = sample(p["vib"],      0,    20)
    pressure = sample(p["pressure"], 0,    15)
    current  = sample(p["current"],  0,    60)
    rpm      = sample(p["rpm"],      0,  4000)

    if failure:
        status = "Failed"
        temp   = round(min(temp + rng.uniform(15, 30), 120), 2)
        vib    = round(min(vib  + rng.uniform(5,  10),  20), 2)
    elif temp > config.THRESHOLDS["temperature_critical"] or vib > config.THRESHOLDS["vibration_critical"]:
        status = "Critical"
    elif temp > config.THRESHOLDS["temperature_high"] or vib > config.THRESHOLDS["vibration_high"]:
        status = "Warning"
    else:
        status = "Running"

    return {
        "timestamp":   timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id":  machine_id,
        "machine_name": config.MACHINES[machine_id]["name"],
        "machine_type": config.MACHINES[machine_id]["type"],
        "location":    config.MACHINES[machine_id]["location"],
        "temperature": temp,
        "vibration":   vib,
        "pressure":    pressure,
        "current":     current,
        "rpm":         rpm,
        "status":      status,
        "anomaly":     int(anomaly or failure),
    }


def generate_historical_data(hours: int = config.HISTORY_HOURS) -> pd.DataFrame:
    """Generate `hours` of historical logs (1 reading / 5 min / machine)."""
    log.info("Generating %d hours of historical data …", hours)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    rows = []
    now  = datetime.now()
    start = now - timedelta(hours=hours)
    ts   = start

    while ts <= now:
        for mid in config.MACHINES:
            anomaly = random.random() < 0.04
            failure = random.random() < 0.01
            rows.append(generate_row(mid, ts, anomaly=anomaly, failure=failure))
        ts += timedelta(minutes=5)

    df = pd.DataFrame(rows)
    log.info("Generated %d rows", len(df))
    return df


# ── SQLite schema & helpers ───────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS machine_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            machine_id   TEXT    NOT NULL,
            machine_name TEXT,
            machine_type TEXT,
            location     TEXT,
            temperature  REAL,
            vibration    REAL,
            pressure     REAL,
            current      REAL,
            rpm          REAL,
            status       TEXT,
            anomaly      INTEGER DEFAULT 0,
            anomaly_score REAL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            machine_id   TEXT    NOT NULL,
            machine_name TEXT,
            alert_type   TEXT,
            severity     TEXT,
            message      TEXT,
            acknowledged INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_logs_ts  ON machine_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_logs_mid ON machine_logs(machine_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", config.DB_PATH)


def seed_db(df: pd.DataFrame):
    """Insert the historical DataFrame into SQLite and CSV."""
    conn = sqlite3.connect(config.DB_PATH)
    df.to_sql("machine_logs", conn, if_exists="append", index=False)
    conn.close()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.CSV_PATH, index=False)
    log.info("Seeded DB and CSV with %d rows", len(df))


def append_log_row(row: dict, anomaly_score: float = 0.0):
    """Append a single sensor row to both SQLite and CSV."""
    row["anomaly_score"] = round(anomaly_score, 4)

    conn = sqlite3.connect(config.DB_PATH)
    pd.DataFrame([row]).to_sql("machine_logs", conn, if_exists="append", index=False)
    conn.close()

    df_row = pd.DataFrame([row])
    header = not os.path.exists(config.CSV_PATH)
    df_row.to_csv(config.CSV_PATH, mode="a", header=header, index=False)


def insert_alert(machine_id: str, alert_type: str, severity: str, message: str):
    """Insert an alert record into SQLite."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """INSERT INTO alerts (timestamp, machine_id, machine_name, alert_type, severity, message)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         machine_id,
         config.MACHINES[machine_id]["name"],
         alert_type, severity, message)
    )
    conn.commit()
    conn.close()


def load_logs(hours: int = 2) -> pd.DataFrame:
    """Load recent logs from SQLite."""
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    conn   = sqlite3.connect(config.DB_PATH)
    df     = pd.read_sql(
        "SELECT * FROM machine_logs WHERE timestamp >= ? ORDER BY timestamp DESC",
        conn, params=(cutoff,)
    )
    conn.close()
    return df


def load_alerts(limit: int = 100) -> pd.DataFrame:
    """Load most recent alerts from SQLite."""
    conn = sqlite3.connect(config.DB_PATH)
    df   = pd.read_sql(
        "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
    df = generate_historical_data()
    seed_db(df)
    print("Data generation complete.")
