"""
Coordinator notification module.
Sends escalation alerts via:
  - Email (SMTP) if SMTP_* env vars are set
  - Webhook POST if ESCALATION_WEBHOOK_URL is set
  - Always prints to console (fallback)

Configure in .env:
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=your@email.com
  SMTP_PASS=yourapppassword
  SMTP_FROM=alerts@yourapp.com
  ESCALATION_WEBHOOK_URL=https://your-server.com/escalation
"""
import os
import json
import smtplib
import urllib.request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_email_alert(patient: dict, triggers: list, report: dict):
    """Send escalation email to coordinator via SMTP."""
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass]):
        print("   ℹ️  SMTP not configured — skipping email alert.")
        return

    to_addr = patient["coordinator_email"]
    subject = f"[ALERT] Patient Follow-Up: {patient['name']} needs attention"

    trigger_lines = "\n".join(f"  - {t.replace('_', ' ').title()}" for t in triggers)
    body = f"""Escalation Alert — Missed-Dose & Follow-Up Companion
=====================================================

Patient: {patient['name']} ({patient['id']})
Phone:   {patient['phone']}
Triggers:
{trigger_lines}

Summary: {report.get('summary', 'N/A')}

Medication Adherence:
{json.dumps(report.get('medication_adherence', {}), indent=2)}

Symptoms Reported: {', '.join(report.get('symptoms_reported', [])) or 'None'}

Appointment Confirmed: {report.get('appointment_confirmed')}

Please follow up with this patient immediately.
"""

    msg = MIMEMultipart()
    msg["From"] = smtp_from
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, to_addr, msg.as_string())
        print(f"   ✉️  Email sent to {to_addr}")
    except Exception as e:
        print(f"   ⚠️  Email failed: {e}")


def send_webhook_alert(patient: dict, triggers: list, report: dict):
    """POST escalation payload to webhook URL."""
    url = os.getenv("ESCALATION_WEBHOOK_URL")
    if not url:
        return

    payload = json.dumps({
        "patient_id": patient["id"],
        "patient_name": patient["name"],
        "patient_phone": patient["phone"],
        "coordinator_email": patient["coordinator_email"],
        "coordinator_phone": patient["coordinator_phone"],
        "triggers": triggers,
        "report": report,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"   🔗 Webhook delivered: HTTP {resp.status}")
    except Exception as e:
        print(f"   ⚠️  Webhook failed: {e}")


def notify(patient: dict, triggers: list, report: dict):
    """Send all configured notifications for an escalation."""
    send_email_alert(patient, triggers, report)
    send_webhook_alert(patient, triggers, report)
