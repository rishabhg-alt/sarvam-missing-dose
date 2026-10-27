"""Escalation engine: evaluates check-in report and alerts the coordinator."""
import json
from datetime import datetime
from pathlib import Path
from companion.config import ESCALATION_TRIGGERS

LOG_FILE = Path(__file__).parent.parent / "escalation_log.json"

ESCALATION_MESSAGES = {
    "missed_critical_dose": "⚠️  MISSED DOSE: Patient has missed a critical medication.",
    "severe_symptom_reported": "🚨 SEVERE SYMPTOM: Patient reported a severe or worsening symptom.",
    "missed_appointment": "📅 MISSED APPOINTMENT: Patient may miss their upcoming appointment.",
    "patient_confused": "🔴 CONFUSION: Patient appeared confused during check-in.",
    "emergency_symptom": "🆘 EMERGENCY: Patient reported an emergency symptom. Call immediately.",
}

def evaluate_and_escalate(patient: dict, report: dict) -> list:
    """Evaluate report for escalation triggers. Returns list of triggered reasons."""
    flags = report.get("escalation_flags", {})
    triggered = [trigger for trigger in ESCALATION_TRIGGERS if flags.get(trigger)]

    if not triggered:
        print(f"\n✅ No escalation needed for {patient['name']}.")
        return []

    print(f"\n{'='*60}")
    print(f"🚨 ESCALATION ALERT — Patient: {patient['name']} ({patient['id']})")
    print(f"   Coordinator: {patient['coordinator_email']}")
    print(f"   Phone: {patient['coordinator_phone']}")
    print(f"{'='*60}")
    for trigger in triggered:
        print(f"   {ESCALATION_MESSAGES.get(trigger, trigger)}")
    print(f"   Summary: {report.get('summary', 'N/A')}")
    print(f"{'='*60}\n")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient["id"],
        "patient_name": patient["name"],
        "coordinator_email": patient["coordinator_email"],
        "coordinator_phone": patient["coordinator_phone"],
        "triggers": triggered,
        "report": report,
    }
    existing = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            existing = json.load(f)
    existing.append(log_entry)
    with open(LOG_FILE, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"   📝 Logged to {LOG_FILE.name}")

    from companion.storage import save_escalation
    from companion.notifier import notify
    save_escalation(patient, triggered, report)
    notify(patient, triggered, report)

    return triggered
