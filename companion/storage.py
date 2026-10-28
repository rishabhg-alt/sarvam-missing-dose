"""
Persistent SQLite storage for patient check-in history and escalation logs.
Replaces the in-memory dict in patient_db.py with durable storage.
Database file: sarvam-demo/checkins.db
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

import os

def _get_db_path() -> Path:
    # If custom disk path is configured and accessible
    custom_dir = os.getenv("RENDER_DISK_PATH")
    if custom_dir:
        p = Path(custom_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p / "checkins.db"
        except Exception:
            pass
    
    # Default to project root
    project_dir = Path(__file__).parent.parent
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        # Test writability
        test_file = project_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return project_dir / "checkins.db"
    except Exception:
        # Fallback to /tmp if project directory is read-only
        tmp_dir = Path("/tmp")
        return tmp_dir / "checkins.db"

DB_PATH = _get_db_path()

_db_initialized = False

def _conn():
    global _db_initialized
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    if not _db_initialized:
        _db_initialized = True
        c.execute("""
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                report_json TEXT NOT NULL,
                escalated INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                triggers_json TEXT NOT NULL,
                coordinator_email TEXT NOT NULL,
                coordinator_phone TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
        """)
        c.commit()
    return c

def init_db():
    """Create tables if they don't exist."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                report_json TEXT NOT NULL,
                escalated INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                triggers_json TEXT NOT NULL,
                coordinator_email TEXT NOT NULL,
                coordinator_phone TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
        """)
        c.commit()

def save_checkin(patient: dict, report: dict, escalated: bool = False):
    with _conn() as c:
        c.execute(
            "INSERT INTO checkins (patient_id, patient_name, timestamp, report_json, escalated) VALUES (?,?,?,?,?)",
            (patient["id"], patient["name"], datetime.now().isoformat(), json.dumps(report, ensure_ascii=False), int(escalated))
        )
        c.commit()

def save_escalation(patient: dict, triggers: list, report: dict):
    with _conn() as c:
        c.execute(
            "INSERT INTO escalations (patient_id, patient_name, timestamp, triggers_json, coordinator_email, coordinator_phone, report_json) VALUES (?,?,?,?,?,?,?)",
            (patient["id"], patient["name"], datetime.now().isoformat(), json.dumps(triggers), patient["coordinator_email"], patient["coordinator_phone"], json.dumps(report, ensure_ascii=False))
        )
        c.commit()

def get_checkin_history(patient_id: str) -> list:
    with _conn() as c:
        rows = c.execute("SELECT * FROM checkins WHERE patient_id=? ORDER BY timestamp DESC", (patient_id,)).fetchall()
        return [{"timestamp": r["timestamp"], "report": json.loads(r["report_json"]), "escalated": bool(r["escalated"])} for r in rows]

def get_all_escalations() -> list:
    with _conn() as c:
        rows = c.execute("SELECT * FROM escalations ORDER BY timestamp DESC").fetchall()
        return [dict(r) for r in rows]
