"""In-memory patient registry. Loads from patients.json."""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

PATIENTS_FILE = Path(__file__).parent.parent / "patients.json"

_patients: dict = {}
_checkin_history: dict = {}

def load_patients():
    global _patients
    with open(PATIENTS_FILE) as f:
        data = json.load(f)
    _patients = {p["id"]: p for p in data}

def get_patient(patient_id: str) -> Optional[dict]:
    if not _patients:
        load_patients()
    return _patients.get(patient_id)

def get_all_patients() -> list:
    if not _patients:
        load_patients()
    return list(_patients.values())

def record_checkin(patient_id: str, checkin_data: dict):
    if patient_id not in _checkin_history:
        _checkin_history[patient_id] = []
    checkin_data["timestamp"] = datetime.now().isoformat()
    _checkin_history[patient_id].append(checkin_data)

def get_checkin_history(patient_id: str) -> list:
    return _checkin_history.get(patient_id, [])
