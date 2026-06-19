# 🏭 AI-Powered Industrial Machine Monitoring & Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A production-grade, end-to-end AI system for real-time monitoring of 10 industrial machines.  
It detects anomalies using **Isolation Forest**, raises threshold-based alerts, persists everything to **SQLite + CSV**, and displays a live **Streamlit** dashboard that auto-refreshes every 5 seconds.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🏭 **10 Industrial Machines** | CNC, Hydraulic Press, Robot Arms, Laser Cutter, Welding Robot … |
| 🤖 **Isolation Forest** | Unsupervised ML anomaly detection on 5 sensor features |
| ⚠️ **Multi-type Alerts** | High temperature, abnormal vibration, machine failure, overspeed, overpressure |
| 🗄️ **Dual persistence** | SQLite + CSV — every reading and alert stored |
| 🔄 **Live Monitor** | Background service polls every 5 seconds |
| 📊 **Streamlit Dashboard** | 7 dashboard sections with Plotly charts, auto-refresh |

---

## 📁 Project Structure

```
ai_maintenance_system/
├── config.py            # Central constants: paths, thresholds, machine list
├── data_generator.py    # Sensor simulation, DB init, helpers
├── model_trainer.py     # Isolation Forest train / load / predict
├── monitor_service.py   # Background monitoring loop (5 s)
├── dashboard.py         # Streamlit dashboard
├── setup.py             # One-shot setup script
├── requirements.txt
├── data/
│   ├── maintenance.db   # SQLite (auto-created)
│   └── machine_logs.csv # CSV export (auto-created)
├── models/
│   ├── isolation_forest.joblib
│   └── scaler.joblib
└── logs/
    └── monitor.log
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/ai-maintenance-system.git
cd ai-maintenance-system
pip install -r requirements.txt
```

### 2. Run setup (once)

```bash
python setup.py
```

This will:
- Create directory structure
- Generate **24 hours** of historical sensor data
- Seed SQLite + CSV
- Train the Isolation Forest model

### 3. Start the monitoring service

```bash
# Terminal 1
python monitor_service.py
```

The service generates one new sensor reading per machine every **5 seconds**, scores it, and writes alerts to the DB.

### 4. Launch the dashboard

```bash
# Terminal 2
streamlit run dashboard.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Dashboard Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Machine Status Cards** | Per-machine live status (Running / Warning / Critical / Failed) |
| 2 | **Fleet Health Overview** | Pie chart + hourly status bar chart |
| 3 | **Active Alerts Table** | Filterable real-time alert feed |
| 4 | **Temperature Trends** | Time-series lines + current bar chart with threshold lines |
| 5 | **Vibration Trends** | Same layout for vibration data |
| 6 | **Failure Statistics** | Failure counts, alert type distribution, worst offenders |
| 7 | **Anomaly History** | Anomaly scatter overlay, per-machine anomaly rate, recent anomaly table |

---

## 🤖 ML Model

- **Algorithm**: `sklearn.ensemble.IsolationForest`
- **Features**: `temperature`, `vibration`, `pressure`, `current`, `rpm`
- **Contamination**: 5% (tunable in `config.py`)
- **Estimators**: 200 trees
- **Scaling**: `StandardScaler` (saved alongside the model)

Retrain at any time:
```bash
python model_trainer.py
```

---

## ⚙️ Configuration

All tuneable parameters live in **`config.py`**:

```python
THRESHOLDS = {
    "temperature_high":    85.0,   # °C  — Warning
    "temperature_critical": 95.0,  # °C  — Critical
    "vibration_high":       7.5,   # mm/s
    "vibration_critical":  12.0,
    "pressure_high":        8.5,   # bar
    "rpm_high":          3200.0,
    "rpm_low":            400.0,
}
MONITOR_INTERVAL_SECONDS = 5
HISTORY_HOURS            = 24
CONTAMINATION            = 0.05
```

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Pandas** — data wrangling
- **Scikit-learn** — Isolation Forest, StandardScaler
- **Streamlit** — dashboard UI
- **Plotly** — interactive charts
- **SQLite** — lightweight persistence
- **Joblib** — model serialisation
- **Schedule** — (optional) cron-style scheduling

---

## 📄 License

MIT © 2024 — free to use, modify, and distribute.
