"""
Persistent SQLite storage for patient check-in history and escalation logs.
Replaces the in-memory dict in patient_db.py with durable storage.
Database file: sarvam-demo/checkins.db
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "checkins.db"

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
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
