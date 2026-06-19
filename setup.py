"""
setup.py
One-shot initialisation script.
Run ONCE before starting the monitor service or dashboard:
    python setup.py
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("═══════════════════════════════════════════════")
    log.info("  AI Industrial Maintenance System — Setup")
    log.info("═══════════════════════════════════════════════")

    # 1. Create directory structure
    import config
    for d in [config.DATA_DIR, config.MODEL_DIR, "logs"]:
        os.makedirs(d, exist_ok=True)
    log.info("✔ Directory structure created")

    # 2. Initialise database
    from data_generator import init_db, generate_historical_data, seed_db
    init_db()
    log.info("✔ SQLite database initialised")

    # 3. Generate historical seed data
    log.info("Generating %dh of historical data …", config.HISTORY_HOURS)
    df = generate_historical_data(hours=config.HISTORY_HOURS)
    seed_db(df)
    log.info("✔ Historical data seeded (%d rows)", len(df))

    # 4. Train Isolation Forest
    from model_trainer import train_model
    model, scaler = train_model(force=True)
    log.info("✔ Isolation Forest trained and saved")

    log.info("═══════════════════════════════════════════════")
    log.info("  Setup complete!  Next steps:")
    log.info("")
    log.info("  Terminal 1 — Start monitor service:")
    log.info("    python monitor_service.py")
    log.info("")
    log.info("  Terminal 2 — Launch dashboard:")
    log.info("    streamlit run dashboard.py")
    log.info("═══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
