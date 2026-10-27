"""
Daily Scheduler — runs check-ins for all patients automatically.

Usage:
    python scheduler.py            # Start the scheduler (blocks)
    python scheduler.py --now      # Run all check-ins immediately and exit

Schedule: runs daily at 10:00 AM local time.
Logs: scheduler.log

Requires:
    pip install apscheduler
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def run_all_checkins():
    """Run a full check-in for every patient in patients.json."""
    log.info("=" * 50)
    log.info(f"Starting daily check-in run at {datetime.now().isoformat()}")

    from companion import patient_db
    from companion.call_simulator import simulate_call
    from companion.storage import init_db

    init_db()
    patient_db.load_patients()
    patients = patient_db.get_all_patients()
    log.info(f"Found {len(patients)} patients to check in")

    for patient in patients:
        pid = patient["id"]
        log.info(f"Calling {patient['name']} ({pid}) — {patient['language']}")
        try:
            simulate_call(pid)
            log.info(f"✅ Check-in complete for {patient['name']}")
        except Exception as e:
            log.error(f"❌ Check-in failed for {patient['name']}: {e}")

    log.info("Daily check-in run complete.")
    log.info("=" * 50)


def main():
    if "--now" in sys.argv:
        run_all_checkins()
        return

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        log.error("apscheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(run_all_checkins, "cron", hour=10, minute=0,
                      id="daily_checkin", name="Daily Patient Check-In")

    log.info("Scheduler started. Daily check-ins at 10:00 AM IST.")
    log.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
