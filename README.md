# Missed-Dose & Follow-Up Companion

Automated outbound voice agent for post-discharge rural patient follow-up, built on [Sarvam AI](https://sarvam.ai).

Calls patients in their native Indian language, conducts a structured medication/symptom/appointment check-in, extracts a structured report, and escalates anomalies to a supervising healthcare coordinator.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your SARVAM_API_KEY (from dashboard.sarvam.ai)

# 3. Run a demo check-in (text-based, no phone needed)
python run_checkup.py P001        # Single patient
python run_checkup.py             # All patients
python run_checkup.py P001 --audio  # With TTS audio playback (macOS)
```

---

## Project Structure

```
sarvam-demo/
├── run_checkup.py          # CLI entry point for manual/demo check-ins
├── scheduler.py            # Daily automated check-in scheduler (APScheduler)
├── patients.json           # Patient registry (id, language, medications, etc.)
├── requirements.txt
├── .env.example
│
├── companion/
│   ├── config.py           # Language codes, voice mappings, model names
│   ├── patient_db.py       # Loads patients.json, tracks in-memory session history
│   ├── conversation.py     # LLM check-in engine (sarvam-105b-conversations)
│   ├── tts.py              # Text-to-Speech wrapper — Bulbul v4-flash
│   ├── stt.py              # Speech-to-Text wrapper — Saaras v4
│   ├── escalation.py       # Evaluates report flags, alerts coordinator
│   ├── notifier.py         # Email (SMTP) + webhook notifications
│   ├── storage.py          # SQLite persistence (checkins.db)
│   └── call_simulator.py   # Simulates a full call session with text turns
│
└── bridge/
    └── exotel_bridge.py    # FastAPI WebSocket server for live Exotel calls
```

---

## How It Works

Every check-in call covers exactly **3 structured areas**, conducted by `sarvam-105b-conversations` in the patient's native language:

1. **Medication Adherence** — asks about each prescribed medication by name
2. **Symptom Check** — asks if any new or worsening symptoms since discharge
3. **Appointment Confirmation** — reminds and confirms the next appointment date

At the end of the call, the LLM produces a structured JSON report:

```json
{
  "medication_adherence": {
    "Metformin 500mg twice daily": "taken",
    "Amlodipine 5mg once daily": "missed"
  },
  "symptoms_reported": ["mild headache"],
  "appointment_confirmed": true,
  "escalation_flags": {
    "missed_critical_dose": true,
    "severe_symptom_reported": false,
    "missed_appointment": false,
    "patient_confused": false,
    "emergency_symptom": false
  },
  "summary": "Patient took Metformin but missed Amlodipine; reports mild headache; confirmed appointment."
}
```

---

## Supported Languages

| Language   | Code    | TTS Voice                  |
|------------|---------|----------------------------|
| Hindi      | `hi-IN` | `ritu_hi_medical`          |
| Tamil      | `ta-IN` | `kavitha_ta_conversation`  |
| Telugu     | `te-IN` | `kavitha_te_conversation`  |
| Bengali    | `bn-IN` | `roopa_bn_conversational`  |
| Marathi    | `mr-IN` | `pooja_mr_conversational`  |
| Gujarati   | `gu-IN` | `pooja_gu_conversational`  |
| Kannada    | `kn-IN` | `kavitha_kn_conversation`  |
| Malayalam  | `ml-IN` | `kavitha_ml_conversation`  |
| English    | `en-IN` | `ishita_en_medical`        |

To add a patient in a new language, add them to `patients.json` with the correct `language` code and the system adapts automatically.

---

## Escalation

The following flags trigger an escalation alert:

| Flag | Meaning |
|------|---------|
| `missed_critical_dose` | Patient missed a prescribed medication |
| `severe_symptom_reported` | Worsening or new severe symptom reported |
| `missed_appointment` | Patient unlikely to attend next appointment |
| `patient_confused` | Patient appeared disoriented during call |
| `emergency_symptom` | Emergency symptom — coordinator called immediately |

When triggered:
- Prints a formatted alert to console with coordinator contact details
- Writes a full record to `escalation_log.json`
- Saves to `checkins.db` (SQLite) with `escalated=true`
- Sends an email alert if SMTP is configured
- POSTs to a webhook if `ESCALATION_WEBHOOK_URL` is set

---

## Persistent Storage

All check-ins and escalations are stored in `checkins.db` (SQLite, auto-created on first run).

```python
from companion.storage import init_db, get_checkin_history, get_all_escalations

init_db()
history = get_checkin_history("P001")   # List of past check-ins for patient
escalations = get_all_escalations()     # All escalation records
```

---

## Daily Scheduler

```bash
# Run all check-ins right now
python scheduler.py --now

# Start the daily scheduler (runs at 10:00 AM IST every day)
python scheduler.py
```

Logs are written to `scheduler.log`.

---

## Real Phone Calls (Exotel)

The `bridge/exotel_bridge.py` is a FastAPI WebSocket server that bridges live Exotel calls through the full Sarvam STT → LLM → TTS pipeline.

### Setup

1. Sign up at [exotel.com](https://exotel.com) and provision a virtual number (ExoPhone)
2. Add Exotel credentials to `.env`:
   ```
   EXOTEL_SID=your_sid
   EXOTEL_API_KEY=your_key
   EXOTEL_API_TOKEN=your_token
   EXOTEL_PHONE_NUMBER=+91XXXXXXXXXX
   PUBLIC_URL=https://your-ngrok-url.ngrok.io
   ```
3. Install and start ngrok (or deploy to a VPS):
   ```bash
   ngrok http 8000
   # Copy the https URL → paste as PUBLIC_URL in .env
   ```
4. Start the bridge:
   ```bash
   pip install fastapi uvicorn httpx
   python bridge/exotel_bridge.py
   ```
5. Trigger an outbound call:
   ```bash
   curl -X POST http://localhost:8000/trigger-call/P001
   ```

The bridge handles the full real-time audio loop:
- Receives 8kHz mulaw audio chunks from Exotel over WebSocket
- Transcribes with Saaras v4 STT
- Gets reply from sarvam-105b-conversations LLM (in patient's language)
- Synthesizes with Bulbul v4-flash TTS at 8kHz
- Streams audio back to Exotel → patient hears it

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SARVAM_API_KEY` | ✅ | Sarvam API key from `dashboard.sarvam.ai` |
| `SMTP_HOST` | Optional | SMTP server for email alerts (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | Optional | SMTP port (default: `587`) |
| `SMTP_USER` | Optional | SMTP login email |
| `SMTP_PASS` | Optional | SMTP password / app password |
| `SMTP_FROM` | Optional | Sender email address |
| `ESCALATION_WEBHOOK_URL` | Optional | URL to POST escalation JSON to |
| `EXOTEL_SID` | For calls | Exotel account SID |
| `EXOTEL_API_KEY` | For calls | Exotel API key |
| `EXOTEL_API_TOKEN` | For calls | Exotel API token |
| `EXOTEL_PHONE_NUMBER` | For calls | Your provisioned Exotel virtual number |
| `PUBLIC_URL` | For calls | Public HTTPS URL of this server (ngrok or VPS) |

---

## Sarvam APIs Used

| API | Model | Purpose |
|-----|-------|---------|
| Chat Completions | `sarvam-105b-conversations` | Drives multilingual structured check-in conversation |
| Text-to-Speech | `bulbul:v4-flash` | Converts agent replies to natural audio in patient's language |
| Speech-to-Text | `saaras:v4` | Transcribes patient speech during live calls |

All APIs use the same `SARVAM_API_KEY`. Base URL: `https://api.sarvam.ai`
