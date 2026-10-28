"""
Missed-Dose & Follow-Up Companion — Web Dashboard
===================================================
Single-file FastAPI server with embedded HTML dashboard.

Run:
    python dashboard.py

Open: http://localhost:8000

Features:
  - Patient list with language, medications, last check-in status
  - Trigger a check-in for any patient (runs in background)
  - Live log streaming via SSE while check-in runs
  - Check-in history per patient with structured report view
  - Escalation log with all alerts
  - REST API for all data
"""

import asyncio
import json
import queue
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from companion import patient_db
from companion.storage import init_db, get_checkin_history, get_all_escalations, save_checkin

# ── App init ─────────────────────────────────────────────────────────────────

# Initialize immediately at startup
try:
    init_db()
    patient_db.load_patients()
except Exception as e:
    print(f"Warning during startup init: {e}")

def _keepalive_worker():
    """
    Automatic keepalive for Render free tier.
    Pings the public external URL every 10 minutes to prevent the 15-minute inactivity spin down.
    """
    import time
    import httpx
    import os

    # Wait 60s for server to complete startup
    time.sleep(60)

    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    if not url and os.getenv("RENDER_EXTERNAL_HOSTNAME"):
        url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}"

    if not url:
        print("ℹ️ Keepalive: No public URL detected yet. Set PUBLIC_URL or RENDER_EXTERNAL_URL.")
        return

    health_url = f"{url.rstrip('/')}/health"
    print(f"⚡ Keepalive active: pinging {health_url} every 10 mins to stay awake 24/7.")

    while True:
        try:
            time.sleep(600)  # 10 minutes (well under the 15 min free-tier cutoff)
            resp = httpx.get(health_url, timeout=30)
            if resp.status_code == 200:
                print(f"💓 Keepalive ping successful: HTTP {resp.status_code}")
            else:
                print(f"⚠️ Keepalive ping returned: HTTP {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Keepalive ping error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    patient_db.load_patients()
    # Start self-keepalive daemon
    t = threading.Thread(target=_keepalive_worker, daemon=True)
    t.start()
    yield

app = FastAPI(title="Missed-Dose Companion", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

# ── SSE event bus ─────────────────────────────────────────────────────────────

_log_queues: list[queue.Queue] = []

def broadcast_event(event_dict: dict):
    payload = json.dumps(event_dict, ensure_ascii=False)
    for q in list(_log_queues):
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass

def broadcast_log(msg: str):
    broadcast_event({"type": "log", "text": msg})


class LogCapture:
    """Captures print() output and broadcasts to SSE clients."""
    def __init__(self, original):
        self.original = original

    def write(self, text):
        self.original.write(text)
        stripped = text.strip()
        if stripped:
            broadcast_log(stripped)

    def flush(self):
        self.original.flush()

    def isatty(self):
        return self.original.isatty()

    def __getattr__(self, name):
        return getattr(self.original, name)


sys.stdout = LogCapture(sys.stdout)

# ── Audio Cache & Active check-in jobs ────────────────────────────────────────

_audio_cache: dict[str, bytes] = {}
_active_jobs: dict[str, str] = {}   # patient_id -> "running" | "done" | "error"


def _run_checkin_thread(patient_id: str):
    """Run an interactive turn-by-turn check-in, streaming both speech turns and reports."""
    _active_jobs[patient_id] = "running"
    try:
        from companion import conversation, patient_db, escalation, notifier, tts
        from companion.config import get_speaker_for_patient, get_speaker_for_agent
        from companion.storage import save_checkin

        patient = patient_db.get_patient(patient_id)
        if not patient:
            broadcast_log(f"[{patient_id}] ❌ Patient not found.")
            _active_jobs[patient_id] = "error"
            return

        lang = patient["language"]
        pat_gender = patient.get("gender", "male")
        agent_speaker = get_speaker_for_agent(lang)
        patient_speaker = get_speaker_for_patient(patient)

        broadcast_event({
            "type": "call_start",
            "patient_id": patient_id,
            "patient_name": patient["name"],
            "phone": patient["phone"],
            "gender": pat_gender,
            "language": lang
        })
        broadcast_log(f"📞 CALL CONNECTED: {patient['name']} ({patient['phone']}) in {lang} [Agent:{agent_speaker}, Patient:{patient_speaker}]")

        report = None
        turn_num = 0

        for item_type, content in conversation.run_turn_by_turn(patient):
            if item_type in ("agent", "patient"):
                turn_num += 1
                role = "agent" if item_type == "agent" else "patient"
                spk = "AI Clinic Assistant" if role == "agent" else patient["name"]
                spk_voice = agent_speaker if role == "agent" else patient_speaker
                turn_gender = "female" if role == "agent" else pat_gender

                # Pre-synthesize TTS audio directly into cache BEFORE broadcasting turn.
                cache_key = f"{spk_voice}:{lang}:{content}"
                if cache_key not in _audio_cache:
                    try:
                        audio_data = tts.speak(content, lang, speaker=spk_voice)
                        if audio_data:
                            _audio_cache[cache_key] = audio_data
                    except Exception as err:
                        print(f"Pre-warm TTS error ({spk_voice}): {err}")

                broadcast_event({
                    "type": "turn",
                    "role": role,
                    "speaker": spk,
                    "speaker_voice": spk_voice,
                    "gender": turn_gender,
                    "text": content,
                    "lang": lang,
                    "turn": turn_num
                })
            elif item_type == "report":
                report = content

        if report:
            triggered = escalation.evaluate_and_escalate(patient, report)
            escalated = bool(triggered)
            save_checkin(patient, report, escalated=escalated)
            if triggered:
                notifier.notify(patient, triggered, report)
            broadcast_event({
                "type": "call_end",
                "patient_id": patient_id,
                "escalated": escalated,
                "report": report
            })
            broadcast_log(f"[{patient_id}] ✅ Call completed. Outcome: {report.get('summary', 'Done')}")
        else:
            broadcast_log(f"[{patient_id}] ⚠️ Call ended without structured report.")

        _active_jobs[patient_id] = "done"
    except Exception as e:
        broadcast_log(f"[{patient_id}] ❌ Call error: {e}")
        _active_jobs[patient_id] = "error"


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/patients")
def api_patients():
    patients = patient_db.get_all_patients()
    result = []
    for p in patients:
        history = get_checkin_history(p["id"])
        last = history[0] if history else None
        result.append({
            **p,
            "total_checkins": len(history),
            "last_checkin": last["timestamp"] if last else None,
            "last_escalated": last["escalated"] if last else False,
            "job_status": _active_jobs.get(p["id"]),
        })
    return result


@app.get("/api/patients/{patient_id}/history")
def api_history(patient_id: str):
    return get_checkin_history(patient_id)


@app.get("/api/escalations")
def api_escalations():
    rows = get_all_escalations()
    result = []
    for r in rows:
        result.append({
            **r,
            "triggers": json.loads(r["triggers_json"]) if isinstance(r["triggers_json"], str) else r["triggers_json"],
            "report": json.loads(r["report_json"]) if isinstance(r["report_json"], str) else r["report_json"],
        })
    return result


@app.post("/api/checkin/{patient_id}")
def api_trigger_checkin(patient_id: str):
    patient = patient_db.get_patient(patient_id)
    if not patient:
        return JSONResponse({"error": f"Patient {patient_id} not found"}, status_code=404)
    if _active_jobs.get(patient_id) == "running":
        return JSONResponse({"error": "Check-in already running for this patient"}, status_code=409)
    t = threading.Thread(target=_run_checkin_thread, args=(patient_id,), daemon=True)
    t.start()
    return {"status": "started", "patient_id": patient_id}


@app.get("/api/speak")
def api_speak(text: str, lang: str = "hi-IN", role: str = "agent", gender: str = "male", speaker: str = None):
    """Synthesizes text to speech using Sarvam Bulbul and returns WAV audio (with in-memory cache)."""
    from companion import tts
    from companion.config import get_speaker_for_agent, VOICE_MAP_FEMALE, VOICE_MAP_MALE
    from fastapi.responses import Response

    if not speaker:
        if role == "agent":
            speaker = get_speaker_for_agent(lang)
        elif str(gender).lower() == "female":
            speaker = VOICE_MAP_FEMALE.get(lang, "kavitha")
        else:
            speaker = VOICE_MAP_MALE.get(lang, "shubh")

    cache_key = f"{speaker}:{lang}:{text}"
    if cache_key in _audio_cache:
        return Response(content=_audio_cache[cache_key], media_type="audio/wav")

    try:
        audio = tts.speak(text, lang, speaker=speaker)
        _audio_cache[cache_key] = audio
        return Response(content=audio, media_type="audio/wav")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/logs")
async def api_logs(request: Request) -> StreamingResponse:
    """SSE endpoint — streams live structured turn events and logs to the browser."""
    q: queue.Queue = queue.Queue(maxsize=200)
    _log_queues.append(q)

    async def generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = q.get_nowait()
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.2)
        finally:
            if q in _log_queues:
                _log_queues.remove(q)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Missed-Dose Companion</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f7fa; color: #1a1a2e; min-height: 100vh; }
  header { background: #1a1a2e; color: white; padding: 16px 32px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; font-weight: 600; }
  header span { font-size: 1.5rem; }
  .subtitle { font-size: 0.8rem; color: #8888aa; margin-top: 2px; }

  nav { display: flex; gap: 0; background: white; border-bottom: 1px solid #e0e0ef; padding: 0 32px; }
  nav button { padding: 12px 20px; border: none; background: none; cursor: pointer;
               font-size: 0.9rem; color: #666; border-bottom: 3px solid transparent;
               transition: all 0.15s; }
  nav button.active { color: #1a1a2e; border-bottom-color: #6c63ff; font-weight: 600; }
  nav button:hover:not(.active) { color: #1a1a2e; background: #f5f7fa; }

  main { padding: 32px; max-width: 1200px; margin: 0 auto; }
  section { display: none; }
  section.active { display: block; }

  h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 20px; color: #1a1a2e; }

  /* Patient grid */
  .patients-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
  .patient-card { background: white; border-radius: 12px; padding: 20px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.08); border: 1px solid #eee; }
  .patient-card .name { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
  .patient-card .meta { font-size: 0.8rem; color: #888; margin-bottom: 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 0.72rem;
           font-weight: 600; margin-right: 4px; }
  .badge.lang { background: #ede9fe; color: #6c63ff; }
  .badge.ok { background: #d1fae5; color: #059669; }
  .badge.alert { background: #fee2e2; color: #dc2626; }
  .badge.running { background: #fef3c7; color: #d97706; }
  .meds { font-size: 0.8rem; color: #555; margin: 10px 0; line-height: 1.6; }
  .meds strong { color: #333; display: block; margin-bottom: 2px; }
  .last-checkin { font-size: 0.76rem; color: #aaa; margin-top: 10px; }
  .btn { padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer;
         font-size: 0.85rem; font-weight: 600; transition: all 0.15s; }
  .btn-primary { background: #6c63ff; color: white; }
  .btn-primary:hover { background: #5a52e0; }
  .btn-primary:disabled { background: #c4c4c4; cursor: not-allowed; }
  .btn-sm { padding: 5px 12px; font-size: 0.78rem; }
  .card-footer { display: flex; justify-content: space-between; align-items: center; margin-top: 14px; }

  /* Table */
  table { width: 100%; border-collapse: collapse; background: white;
          border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  th { text-align: left; padding: 12px 16px; font-size: 0.78rem; font-weight: 600;
       color: #888; background: #fafafa; border-bottom: 1px solid #eee; text-transform: uppercase; letter-spacing: 0.05em; }
  td { padding: 12px 16px; font-size: 0.85rem; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafbff; }

  /* Log console */
  #log-console { background: #1a1a2e; color: #a8ff78; font-family: 'SF Mono', 'Fira Code', monospace;
                 font-size: 0.8rem; padding: 20px; border-radius: 12px; height: 400px;
                 overflow-y: auto; line-height: 1.8; white-space: pre-wrap; }
  #log-console .log-line { animation: fadein 0.2s ease; }
  @keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }

  /* Report modal */
  .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
                   z-index: 100; align-items: center; justify-content: center; }
  .modal-overlay.open { display: flex; }
  .modal { background: white; border-radius: 16px; padding: 28px; max-width: 600px;
           width: 90%; max-height: 80vh; overflow-y: auto; position: relative; }
  .modal h3 { font-size: 1rem; font-weight: 700; margin-bottom: 16px; }
  .modal-close { position: absolute; top: 16px; right: 16px; border: none; background: none;
                 font-size: 1.2rem; cursor: pointer; color: #888; }
  .report-section { margin-bottom: 16px; }
  .report-section label { font-size: 0.75rem; font-weight: 700; color: #888;
                          text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 6px; }
  .med-row { display: flex; justify-content: space-between; padding: 6px 10px;
             border-radius: 6px; background: #f5f7fa; margin-bottom: 4px; font-size: 0.84rem; }
  .med-status.taken { color: #059669; font-weight: 600; }
  .med-status.missed { color: #dc2626; font-weight: 600; }
  .med-status.partial { color: #d97706; font-weight: 600; }
  .med-status.unknown { color: #888; }
  .flag-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .symptom-tag { background: #fee2e2; color: #dc2626; border-radius: 99px;
                 padding: 2px 10px; font-size: 0.78rem; }
  .summary-box { background: #f0f4ff; border-radius: 8px; padding: 12px;
                 font-size: 0.85rem; color: #444; line-height: 1.5; }

  .empty { color: #aaa; font-size: 0.85rem; padding: 40px; text-align: center; }
  .stats-row { display: flex; gap: 16px; margin-bottom: 24px; }
  .stat-card { background: white; border-radius: 12px; padding: 16px 24px;
               flex: 1; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .stat-card .num { font-size: 2rem; font-weight: 700; color: #6c63ff; }
  .stat-card .lbl { font-size: 0.8rem; color: #888; margin-top: 2px; }

  /* Interactive Call Bubbles */
  .bubble-wrap { display: flex; flex-direction: column; max-width: 82%; animation: fadein 0.25s ease; margin-bottom: 12px; }
  .bubble-wrap.agent { align-self: flex-start; }
  .bubble-wrap.patient { align-self: flex-end; }
  .bubble-sender { font-size: 0.74rem; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
  .bubble-wrap.agent .bubble-sender { color: #4f46e5; }
  .bubble-wrap.patient .bubble-sender { justify-content: flex-end; color: #059669; }
  .bubble-card { padding: 12px 16px; border-radius: 14px; font-size: 0.92rem; line-height: 1.5; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .bubble-wrap.agent .bubble-card { background: white; border: 1px solid #e0e7ff; color: #1e1b4b; border-bottom-left-radius: 4px; }
  .bubble-wrap.patient .bubble-card { background: #ecfdf5; border: 1px solid #a7f3d0; color: #064e3b; border-bottom-right-radius: 4px; }
  .bubble-action { margin-top: 6px; display: flex; gap: 8px; font-size: 0.72rem; }
  .bubble-wrap.patient .bubble-action { justify-content: flex-end; }
  .play-turn-btn { background: none; border: 1px solid #cbd5e1; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; cursor: pointer; color: #475569; display: flex; align-items: center; gap: 4px; }
  .play-turn-btn:hover { background: #f1f5f9; color: #0f172a; }

  /* Pulsing live dot */
  .live-dot { width: 10px; height: 10px; background: #10b981; border-radius: 50%; display: inline-block; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }

  /* Real-time word streaming cursor */
  .typing-cursor { display: inline-block; color: #4f46e5; font-weight: 800; margin-left: 2px; animation: blink 0.75s infinite; }
  .bubble-wrap.patient .typing-cursor { color: #059669; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
</style>
</head>
<body>

<header>
  <span>🏥</span>
  <div>
    <h1>Missed-Dose & Follow-Up Companion</h1>
    <div class="subtitle">Automated outbound voice check-ins · Sarvam AI</div>
  </div>
</header>

<nav>
  <button id="nav-patients" class="active" onclick="showTab('patients')">👥 Patients</button>
  <button id="nav-escalations" onclick="showTab('escalations')">🚨 Escalations</button>
  <button id="nav-logs" onclick="showTab('logs')">🖥 Live Logs</button>
</nav>

<main>

  <!-- Patients tab -->
  <section id="tab-patients" class="active">
    <div class="stats-row" id="stats-row"></div>
    <h2>Patient Registry</h2>
    <div class="patients-grid" id="patient-grid">
      <div class="empty">Loading patients…</div>
    </div>
  </section>

  <!-- Escalations tab -->
  <section id="tab-escalations">
    <h2>Escalation Log</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Patient</th><th>Triggers</th><th>Coordinator</th><th>Summary</th><th></th>
        </tr>
      </thead>
      <tbody id="escalation-tbody">
        <tr><td colspan="6" class="empty">Loading…</td></tr>
      </tbody>
    </table>
  </section>

  <!-- Logs tab -->
  <section id="tab-logs">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px">
      <h2>📞 Live Outbound Call Session</h2>
      <div style="display:flex; gap:10px; align-items:center">
        <button id="btn-audio-toggle" class="btn btn-sm" style="background:#e0e7ff; color:#4338ca" onclick="toggleAudio()">
          🔊 Voice Audio: ON
        </button>
        <button class="btn btn-sm" style="background:#f3f4f6; color:#4b5563" onclick="clearLiveCall()">
          🗑 Clear Transcript
        </button>
      </div>
    </div>

    <!-- Active Call Banner -->
    <div id="call-banner" style="display:none; background:white; border-radius:12px; padding:16px 20px; border-left:5px solid #10b981; box-shadow:0 1px 4px rgba(0,0,0,0.06); margin-bottom:16px">
      <div style="display:flex; justify-content:space-between; align-items:center">
        <div>
          <div style="font-weight:700; font-size:1rem" id="call-patient-name">Active Call: Ramesh Kumar</div>
          <div style="font-size:0.8rem; color:#6b7280; margin-top:2px" id="call-details">Language: Hindi (hi-IN) · Phone: +919876543210</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px">
          <span class="live-dot"></span>
          <span style="font-size:0.8rem; font-weight:600; color:#059669" id="call-status-badge">Call in Progress</span>
        </div>
      </div>
    </div>

    <!-- Audio Player Indicator -->
    <div id="audio-playing-indicator" style="display:none; background:#ede9fe; color:#6d28d9; padding:8px 16px; border-radius:8px; font-size:0.82rem; font-weight:600; margin-bottom:14px; align-items:center; gap:8px">
      <span>🔊 Speaking:</span>
      <span id="audio-speaking-text" style="font-weight:400; font-style:italic">...</span>
    </div>

    <!-- Dedicated Persistent Audio Elements for Live Conversation and Replay -->
    <audio id="call-audio" preload="auto" style="display:none"></audio>
    <audio id="replay-audio" preload="auto" style="display:none"></audio>

    <!-- Live Speech Bubbles -->
    <div id="call-transcript" style="display:flex; flex-direction:column; margin-bottom:20px; min-height:120px">
      <div class="empty" id="transcript-empty">
        Ready for outbound call.<br>
        Click <strong>📞 Check In</strong> on any patient to start a live voice conversation in their native language!
      </div>
    </div>

    <!-- Raw Logs Collapsible -->
    <details style="background:white; border-radius:10px; border:1px solid #e5e7eb; padding:12px">
      <summary style="font-size:0.82rem; font-weight:600; color:#6b7280; cursor:pointer">Raw Technical & Escalation Logs</summary>
      <div id="log-console" style="margin-top:10px; background:#1a1a2e; color:#a8ff78; font-family:'SF Mono',monospace; font-size:0.75rem; padding:12px; border-radius:8px; height:180px; overflow-y:auto; line-height:1.6; white-space:pre-wrap">Waiting for events…</div>
    </details>
  </section>

</main>

<!-- Report modal -->
<div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
  <div class="modal" id="modal-content">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div id="modal-body"></div>
  </div>
</div>

<script>
const LANG_NAMES = {
  'hi-IN':'Hindi','ta-IN':'Tamil','te-IN':'Telugu','bn-IN':'Bengali',
  'mr-IN':'Marathi','gu-IN':'Gujarati','kn-IN':'Kannada','ml-IN':'Malayalam',
  'pa-IN':'Punjabi','en-IN':'English'
};

let patients = [];
let escalations = [];

// ── Tab switching ────────────────────────────────────────────────────────────
function showTab(name) {
  try {
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    
    const targetSection = document.getElementById('tab-' + name);
    if (targetSection) targetSection.classList.add('active');
    
    const targetBtn = document.getElementById('nav-' + name);
    if (targetBtn) targetBtn.classList.add('active');
    
    if (name === 'escalations') {
      loadEscalations();
    } else if (name === 'patients') {
      loadPatients();
    }
  } catch (err) {
    console.error('Error switching tab:', err);
  }
}

// ── Patients ─────────────────────────────────────────────────────────────────
async function loadPatients() {
  const grid = document.getElementById('patient-grid');
  try {
    const res = await fetch('/api/patients');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    patients = await res.json();
    renderStats();
    renderPatients();
  } catch (err) {
    console.error('Failed to load patients:', err);
    if (grid && !patients.length) {
      grid.innerHTML = '<div class="empty" style="color:#dc2626">Failed to load patients (' + err.message + '). <button class="btn btn-sm btn-primary" onclick="loadPatients()" style="margin-left:8px">Retry</button></div>';
    }
  }
}

function renderStats() {
  const total = patients.length;
  const escalated = patients.filter(p => p.last_escalated).length;
  const checkedIn = patients.filter(p => p.total_checkins > 0).length;
  const statsRow = document.getElementById('stats-row');
  if (statsRow) {
    statsRow.innerHTML = `
      <div class="stat-card"><div class="num">${total}</div><div class="lbl">Total Patients</div></div>
      <div class="stat-card"><div class="num">${checkedIn}</div><div class="lbl">Checked In</div></div>
      <div class="stat-card"><div class="num" style="color:#dc2626">${escalated}</div><div class="lbl">Escalated (last call)</div></div>
    `;
  }
}

function renderPatients() {
  const grid = document.getElementById('patient-grid');
  if (!grid) return;
  if (!patients.length) { grid.innerHTML = '<div class="empty">No patients found in database.</div>'; return; }
  grid.innerHTML = patients.map(p => {
    const lang = LANG_NAMES[p.language] || p.language;
    const lastTime = p.last_checkin ? new Date(p.last_checkin).toLocaleString('en-IN') : 'Never';
    const statusBadge = p.job_status === 'running'
      ? '<span class="badge running">⏳ Running</span>'
      : p.last_escalated
        ? '<span class="badge alert">🚨 Escalated</span>'
        : p.total_checkins > 0
          ? '<span class="badge ok">✅ OK</span>'
          : '';
    const meds = (p.medications || []).map(m => `<span>💊 ${m}</span>`).join('<br>');
    const btnDisabled = p.job_status === 'running' ? 'disabled' : '';
    return `
    <div class="patient-card" id="card-${p.id}">
      <div class="name">${p.name}</div>
      <div class="meta">
        <span class="badge lang">${lang}</span>
        ${statusBadge}
      </div>
      <div class="meds"><strong>Medications</strong>${meds}</div>
      <div class="meds"><strong>Next Appointment</strong>${p.next_appointment}</div>
      <div class="last-checkin">Last check-in: ${lastTime} · ${p.total_checkins} total</div>
      <div class="card-footer">
        <button class="btn btn-primary btn-sm" id="btn-${p.id}" onclick="triggerCheckin('${p.id}')" ${btnDisabled}>
          📞 Check In
        </button>
        ${p.total_checkins > 0 ? `<button class="btn btn-sm" style="background:#f0f4ff;color:#6c63ff" onclick="showHistory('${p.id}')">📋 History</button>` : ''}
      </div>
    </div>`;
  }).join('');
}

async function triggerCheckin(patientId) {
  unlockAudio();
  const btn = document.getElementById('btn-' + patientId);
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Calling…';
  }

  // Switch to logs tab
  showTab('logs');
  clearLiveCall();

  const patient = patients.find(p => p.id === patientId);
  if (patient) {
    const banner = document.getElementById('call-banner');
    const pName = document.getElementById('call-patient-name');
    const pDetails = document.getElementById('call-details');
    const pBadge = document.getElementById('call-status-badge');
    if (banner && pName && pDetails && pBadge) {
      banner.style.display = 'block';
      pName.textContent = 'Connecting Outbound Call: ' + patient.name;
      const lang = LANG_NAMES[patient.language] || patient.language;
      pDetails.textContent = 'Native Language: ' + lang + ' · Phone: ' + patient.phone;
      pBadge.textContent = 'Dialing...';
    }
  }

  try {
    await fetch('/api/checkin/' + patientId, { method: 'POST' });
  } catch (err) {
    console.error('Trigger checkin failed:', err);
  }

  // Poll for completion
  const poll = setInterval(async () => {
    await loadPatients();
    const p = patients.find(x => x.id === patientId);
    if (p && p.job_status !== 'running') {
      clearInterval(poll);
      if (btn) {
        btn.disabled = false;
        btn.textContent = '📞 Check In';
      }
    }
  }, 3000);
}

// ── Audio & Dynamic Word Streaming Engine ────────────────────────────────────
let turnQueue = [];
let isPlayingTurn = false;
let audioEnabled = true;
let audioUnlocked = false;
let currentTurn = null;

let callAudio = null;
let replayAudio = null;

function getCallAudio() {
  if (!callAudio) callAudio = document.getElementById('call-audio');
  return callAudio;
}

function getReplayAudio() {
  if (!replayAudio) replayAudio = document.getElementById('replay-audio');
  return replayAudio;
}

function updateAudioButton() {
  const btn = document.getElementById('btn-audio-toggle');
  if (btn) {
    btn.innerHTML = audioEnabled ? '🔊 Voice Audio: ON' : '🔈 Voice Audio: MUTED';
    btn.style.background = audioEnabled ? '#e0e7ff' : '#fee2e2';
    btn.style.color = audioEnabled ? '#4338ca' : '#dc2626';
  }
}

function toggleAudio() {
  unlockAudio();
  audioEnabled = !audioEnabled;
  updateAudioButton();

  const cAudio = getCallAudio();
  const rAudio = getReplayAudio();

  if (!audioEnabled) {
    // Muting: immediately silence both audio channels
    if (cAudio) {
      cAudio.muted = true;
      cAudio.volume = 0;
    }
    if (rAudio) {
      rAudio.muted = true;
      rAudio.volume = 0;
    }
  } else {
    // Unmuting: immediately restore voice and guarantee sound on current speaker
    if (cAudio) {
      cAudio.muted = false;
      cAudio.volume = 1.0;
      if (isPlayingTurn && currentTurn) {
        try {
          const audioUrl = '/api/speak?role=' + currentTurn.role + '&lang=' + encodeURIComponent(currentTurn.lang) + '&speaker=' + encodeURIComponent(currentTurn.speaker_voice || '') + '&gender=' + encodeURIComponent(currentTurn.gender || 'female') + '&text=' + encodeURIComponent(currentTurn.text);
          // If audio reached the end while muted or was not loaded, reload and play immediately!
          if (cAudio.ended || cAudio.paused || isNaN(cAudio.duration) || cAudio.currentTime >= (cAudio.duration - 0.2)) {
            cAudio.src = audioUrl;
          }
          cAudio.play().catch(e => console.warn('Unmute play error:', e));
        } catch (err) {
          console.warn('Error in unmute play:', err);
        }
      } else if (!isPlayingTurn && turnQueue.length) {
        playNextTurn();
      }
    }

    if (rAudio) {
      rAudio.muted = false;
      rAudio.volume = 1.0;
      if (rAudio.paused && rAudio.src && !rAudio.ended) {
        rAudio.play().catch(e => console.warn('Replay unmute error:', e));
      }
    }
  }
}

function unlockAudio() {
  if (!audioUnlocked) {
    try {
      const silent = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
      const c = getCallAudio();
      if (c) {
        c.src = silent;
        c.play().then(() => {
          c.pause();
          c.currentTime = 0;
          audioUnlocked = true;
        }).catch(() => {});
      }
      const r = getReplayAudio();
      if (r) {
        r.src = silent;
        r.play().then(() => {
          r.pause();
          r.currentTime = 0;
        }).catch(() => {});
      }
    } catch (e) {}
  }
}

function enqueueTurn(turn) {
  // Pre-load audio into browser cache with exact speaker voice and gender
  const audioUrl = '/api/speak?role=' + turn.role + '&lang=' + encodeURIComponent(turn.lang) + '&speaker=' + encodeURIComponent(turn.speaker_voice || '') + '&gender=' + encodeURIComponent(turn.gender || 'female') + '&text=' + encodeURIComponent(turn.text);
  const preloaded = new Audio(audioUrl);
  preloaded.preload = "auto";
  preloaded.load();

  turnQueue.push(turn);
  if (!isPlayingTurn) {
    playNextTurn();
  }
}

function createSpeechBubbleSkeleton(role, speaker, lang, fullText, speakerVoice, gender) {
  const transcript = document.getElementById('call-transcript');
  if (!transcript) return null;
  const empty = document.getElementById('transcript-empty');
  if (empty) empty.remove();

  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap ' + (role === 'agent' ? 'agent' : 'patient');

  const sender = document.createElement('div');
  sender.className = 'bubble-sender';
  sender.innerHTML = (role === 'agent' ? '🤖 ' : '👤 ') + speaker;

  const card = document.createElement('div');
  card.className = 'bubble-card';
  // Subtle initial loading placeholder until voice starts
  card.innerHTML = '<span class="voice-loading" style="color:#9ca3af;font-size:0.85rem;font-style:italic">🔊 Connecting voice…</span>';

  const action = document.createElement('div');
  action.className = 'bubble-action';
  action.style.display = 'none'; // revealed once turn finishes speaking

  const playBtn = document.createElement('button');
  playBtn.className = 'play-turn-btn';
  playBtn.innerHTML = '🔊 Replay Audio';
  playBtn.onclick = () => playSingleTurn(role, lang, fullText, speakerVoice, gender);
  action.appendChild(playBtn);

  wrap.appendChild(sender);
  wrap.appendChild(card);
  wrap.appendChild(action);

  transcript.appendChild(wrap);
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

  return { wrap, card, action };
}

let pendingCallEnd = null;

function applyCallEnd(data) {
  const pBadge = document.getElementById('call-status-badge');
  if (pBadge) {
    pBadge.textContent = data.escalated ? 'Call Ended · 🚨 Escalated' : 'Call Ended · ✅ Adherent';
    pBadge.style.color = data.escalated ? '#dc2626' : '#059669';
  }

  const transcript = document.getElementById('call-transcript');
  if (transcript && !document.getElementById('call-conclusion-card')) {
    const summaryCard = document.createElement('div');
    summaryCard.id = 'call-conclusion-card';
    summaryCard.style.marginTop = '20px';
    summaryCard.style.padding = '16px';
    summaryCard.style.borderRadius = '12px';
    summaryCard.style.background = data.escalated ? '#fff1f2' : '#f0fdf4';
    summaryCard.style.border = '2px solid ' + (data.escalated ? '#fca5a5' : '#86efac');

    const heading = document.createElement('div');
    heading.style.fontWeight = '700';
    heading.style.fontSize = '0.95rem';
    heading.style.color = data.escalated ? '#991b1b' : '#166534';
    heading.style.marginBottom = '6px';
    heading.textContent = data.escalated ? '🚨 Call Concluded — Escalation Triggered' : '✅ Call Concluded — Routine Follow-Up Complete';

    const summaryText = document.createElement('div');
    summaryText.style.fontSize = '0.86rem';
    summaryText.style.color = '#374151';
    summaryText.style.marginBottom = '12px';
    summaryText.textContent = data.report ? (data.report.summary || 'Check-in completed successfully.') : 'Check-in completed.';

    const histBtn = document.createElement('button');
    histBtn.className = 'btn btn-sm btn-primary';
    histBtn.textContent = '📋 View Full Structured Report';
    histBtn.onclick = () => showHistory(data.patient_id);

    summaryCard.appendChild(heading);
    summaryCard.appendChild(summaryText);
    summaryCard.appendChild(histBtn);
    transcript.appendChild(summaryCard);
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  }

  loadPatients();
}

function playNextTurn() {
  if (!turnQueue.length) {
    isPlayingTurn = false;
    currentTurn = null;
    const ind = document.getElementById('audio-playing-indicator');
    if (ind) ind.style.display = 'none';

    // When all speech turns have completely finished, apply pending call conclusion!
    if (pendingCallEnd) {
      applyCallEnd(pendingCallEnd);
      pendingCallEnd = null;
    }
    return;
  }

  isPlayingTurn = true;
  const turn = turnQueue.shift();
  currentTurn = turn;

  const bubbleElements = createSpeechBubbleSkeleton(turn.role, turn.speaker, turn.lang, turn.text, turn.speaker_voice, turn.gender);

  const ind = document.getElementById('audio-playing-indicator');
  const indText = document.getElementById('audio-speaking-text');
  if (ind && indText) {
    ind.style.display = 'flex';
    indText.textContent = (turn.role === 'agent' ? '🤖 Clinic Assistant' : '👤 Patient') + ' is speaking…';
  }

  let turnFinished = false;
  let wordStreamTimer = null;
  let audioStarted = false;

  function advanceToNextTurn() {
    if (turnFinished) return;
    turnFinished = true;
    if (wordStreamTimer) clearInterval(wordStreamTimer);
    if (bubbleElements) {
      if (bubbleElements.card) bubbleElements.card.textContent = turn.text;
      if (bubbleElements.action) bubbleElements.action.style.display = 'flex';
    }
    currentTurn = null;
    // Natural slight conversational gap before the next turn
    setTimeout(playNextTurn, 100);
  }

  function startWordStreaming(durationSec) {
    if (turnFinished) return;
    const words = turn.text.split(' ');
    // Match word streaming speed precisely to audio duration
    const totalDurationMs = (durationSec && durationSec > 0.4) ? (durationSec * 1000) : (words.length * 220);
    const msPerWord = Math.max(70, Math.floor(totalDurationMs / words.length));

    let wordIdx = 0;
    if (bubbleElements && bubbleElements.card) {
      bubbleElements.card.textContent = ''; // Clear placeholder hint
      const cursor = document.createElement('span');
      cursor.className = 'typing-cursor';
      cursor.textContent = ' ▌';
      bubbleElements.card.appendChild(cursor);

      wordStreamTimer = setInterval(() => {
        if (wordIdx < words.length) {
          const wordText = (wordIdx > 0 ? ' ' : '') + words[wordIdx];
          cursor.before(wordText);
          wordIdx++;
          window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        } else {
          clearInterval(wordStreamTimer);
          cursor.remove();
          bubbleElements.card.textContent = turn.text;
        }
      }, msPerWord);
    }
  }

  const cAudio = getCallAudio();
  const audioUrl = '/api/speak?role=' + turn.role + '&lang=' + encodeURIComponent(turn.lang) + '&speaker=' + encodeURIComponent(turn.speaker_voice || '') + '&gender=' + encodeURIComponent(turn.gender || 'female') + '&text=' + encodeURIComponent(turn.text);

  // Reset listeners
  cAudio.onplaying = null;
  cAudio.onended = null;
  cAudio.onerror = null;

  cAudio.muted = !audioEnabled;
  cAudio.volume = audioEnabled ? 1 : 0;
  cAudio.src = audioUrl;

  cAudio.onplaying = () => {
    if (!audioStarted) {
      audioStarted = true;
      startWordStreaming(cAudio.duration);
    }
  };

  cAudio.onended = () => {
    advanceToNextTurn();
  };

  cAudio.onerror = (e) => {
    console.warn('Live call audio playback error, streaming words visually:', e);
    if (!audioStarted) {
      audioStarted = true;
      startWordStreaming(turn.text.split(' ').length * 0.22);
    }
    setTimeout(advanceToNextTurn, (turn.text.split(' ').length * 220) + 200);
  };

  cAudio.play().catch(e => {
    console.warn('Call audio play error, streaming visually:', e);
    if (!audioStarted) {
      audioStarted = true;
      startWordStreaming(turn.text.split(' ').length * 0.22);
    }
    setTimeout(advanceToNextTurn, (turn.text.split(' ').length * 220) + 200);
  });

  // Safety timeout: only if audio fails to play after 6 seconds, fallback to visual stream
  setTimeout(() => {
    if (!audioStarted && !turnFinished) {
      audioStarted = true;
      startWordStreaming(turn.text.split(' ').length * 0.22);
      setTimeout(advanceToNextTurn, (turn.text.split(' ').length * 220) + 200);
    }
  }, 6000);
}

function playSingleTurn(role, lang, text, speakerVoice, gender) {
  unlockAudio();
  const rAudio = getReplayAudio();
  if (!rAudio) return;

  // If muted, automatically unmute so the user can hear the replay
  if (!audioEnabled) {
    audioEnabled = true;
    updateAudioButton();
    const cAudio = getCallAudio();
    if (cAudio) {
      cAudio.muted = false;
      cAudio.volume = 1;
    }
  }

  const audioUrl = '/api/speak?role=' + role + '&lang=' + encodeURIComponent(lang) + '&speaker=' + encodeURIComponent(speakerVoice || '') + '&gender=' + encodeURIComponent(gender || 'female') + '&text=' + encodeURIComponent(text);
  rAudio.muted = !audioEnabled;
  rAudio.volume = audioEnabled ? 1 : 0;
  rAudio.src = audioUrl;
  rAudio.play().catch(e => console.warn('Replay play error:', e));
}

function clearLiveCall() {
  const cAudio = getCallAudio();
  if (cAudio) {
    try {
      cAudio.pause();
      cAudio.currentTime = 0;
      cAudio.src = '';
    } catch (e) {}
  }
  const rAudio = getReplayAudio();
  if (rAudio) {
    try {
      rAudio.pause();
      rAudio.currentTime = 0;
      rAudio.src = '';
    } catch (e) {}
  }
  const transcript = document.getElementById('call-transcript');
  if (transcript) {
    transcript.innerHTML = '<div class="empty" id="transcript-empty">Ready for outbound call.<br>Click <strong>📞 Check In</strong> on any patient to start a live voice conversation in their native language!</div>';
  }
  const banner = document.getElementById('call-banner');
  if (banner) banner.style.display = 'none';
  const ind = document.getElementById('audio-playing-indicator');
  if (ind) ind.style.display = 'none';
  turnQueue = [];
  isPlayingTurn = false;
  currentTurn = null;
}

// Keep Render free-tier container active 24/7 while browser is open
setInterval(() => {
  fetch('/health').catch(() => {});
}, 180000);

// ── History modal ─────────────────────────────────────────────────────────────
async function showHistory(patientId) {
  try {
    const res = await fetch('/api/patients/' + patientId + '/history');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const history = await res.json();
    const patient = patients.find(p => p.id === patientId);
    if (!history.length) { alert('No check-in history yet.'); return; }
    renderModal(patient, history[0]);
  } catch (err) {
    alert('Failed to load history: ' + err.message);
  }
}

function renderModal(patient, checkin) {
  const r = checkin.report || {};
  const meds = Object.entries(r.medication_adherence || {}).map(([med, status]) =>
    `<div class="med-row"><span>${med}</span><span class="med-status ${status}">${status.toUpperCase()}</span></div>`
  ).join('');
  const symptoms = (r.symptoms_reported || []).length
    ? r.symptoms_reported.map(s => `<span class="symptom-tag">${s}</span>`).join('')
    : '<span style="color:#aaa;font-size:0.85rem">None reported</span>';
  const flags = Object.entries(r.escalation_flags || {})
    .filter(([,v]) => v)
    .map(([k]) => `<span class="badge alert">${k.replace(/_/g,' ')}</span>`)
    .join('') || '<span style="color:#aaa;font-size:0.85rem">None</span>';

  document.getElementById('modal-body').innerHTML = `
    <h3>📋 Check-In Report — ${patient ? patient.name : 'Patient'}</h3>
    <div style="font-size:0.78rem;color:#aaa;margin-bottom:16px">${new Date(checkin.timestamp).toLocaleString('en-IN')}</div>
    <div class="report-section"><label>Medication Adherence</label>${meds || '<span style="color:#aaa;font-size:0.85rem">None recorded</span>'}</div>
    <div class="report-section"><label>Symptoms Reported</label><div class="flag-list">${symptoms}</div></div>
    <div class="report-section"><label>Appointment Confirmed</label>
      <span class="badge ${r.appointment_confirmed ? 'ok' : 'alert'}">${r.appointment_confirmed ? 'Yes' : 'No / Unknown'}</span>
    </div>
    <div class="report-section"><label>Escalation Flags</label><div class="flag-list">${flags}</div></div>
    <div class="report-section"><label>Summary</label><div class="summary-box">${r.summary || '—'}</div></div>
  `;
  document.getElementById('modal-overlay').classList.add('open');
}

function closeModal(e) {
  if (!e || e.target === document.getElementById('modal-overlay')) {
    document.getElementById('modal-overlay').classList.remove('open');
  }
}

// ── Escalations ───────────────────────────────────────────────────────────────
async function loadEscalations() {
  const tbody = document.getElementById('escalation-tbody');
  try {
    const res = await fetch('/api/escalations');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    escalations = await res.json();
    if (!tbody) return;
    if (!escalations.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">No escalations yet — all clear! 🎉</td></tr>';
      return;
    }
    tbody.innerHTML = escalations.map((e, index) => {
      const triggers = (Array.isArray(e.triggers) ? e.triggers : JSON.parse(e.triggers || '[]'))
        .map(t => `<span class="badge alert">${t.replace(/_/g,' ')}</span>`).join(' ');
      const report = typeof e.report === 'string' ? JSON.parse(e.report) : (e.report || {});
      return `<tr>
        <td style="white-space:nowrap;color:#888">${new Date(e.timestamp).toLocaleString('en-IN')}</td>
        <td><strong>${e.patient_name}</strong><br><span style="font-size:0.76rem;color:#aaa">${e.patient_id}</span></td>
        <td>${triggers}</td>
        <td style="font-size:0.8rem">${e.coordinator_email}<br>${e.coordinator_phone}</td>
        <td style="font-size:0.82rem;color:#555;max-width:220px">${report.summary || '—'}</td>
        <td><button class="btn btn-sm" style="background:#f0f4ff;color:#6c63ff"
            onclick="showEscalationDetail(${index})">View</button></td>
      </tr>`;
    }).join('');
  } catch (err) {
    console.error('Failed to load escalations:', err);
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty" style="color:#dc2626">Failed to load escalations: ' + err.message + '</td></tr>';
    }
  }
}

function showEscalationDetail(index) {
  const e = escalations[index];
  if (!e) return;
  const report = typeof e.report === 'string' ? JSON.parse(e.report) : (e.report || {});
  const patient = { id: e.patient_id, name: e.patient_name };
  renderModal(patient, { report, timestamp: e.timestamp });
}

// ── Live Event Stream (SSE) ────────────────────────────────────────────────────
const logConsole = document.getElementById('log-console');
try {
  const evtSource = new EventSource('/api/logs');
  evtSource.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data);
      if (typeof parsed === 'object' && parsed !== null) {
        if (parsed.type === 'call_start') {
          const banner = document.getElementById('call-banner');
          const pName = document.getElementById('call-patient-name');
          const pDetails = document.getElementById('call-details');
          const pBadge = document.getElementById('call-status-badge');
          if (banner) banner.style.display = 'block';
          if (pName) pName.textContent = 'Active Call: ' + parsed.patient_name;
          if (pDetails) pDetails.textContent = 'Language: ' + (LANG_NAMES[parsed.language] || parsed.language) + ' · Phone: ' + parsed.phone;
          if (pBadge) pBadge.textContent = 'Connected (Live)';
        } else if (parsed.type === 'turn') {
          enqueueTurn(parsed);
        } else if (parsed.type === 'call_end') {
          pendingCallEnd = parsed;
          // Apply immediately ONLY if no turns are playing or waiting in queue
          if (!isPlayingTurn && !turnQueue.length) {
            applyCallEnd(pendingCallEnd);
            pendingCallEnd = null;
          }
        } else if (parsed.type === 'log') {
          if (logConsole) {
            const line = document.createElement('div');
            line.className = 'log-line';
            line.textContent = parsed.text;
            logConsole.appendChild(line);
            logConsole.scrollTop = logConsole.scrollHeight;
          }
        }
      } else {
        if (logConsole) {
          const line = document.createElement('div');
          line.className = 'log-line';
          line.textContent = String(parsed);
          logConsole.appendChild(line);
          logConsole.scrollTop = logConsole.scrollHeight;
        }
      }
    } catch (err) {
      if (logConsole) {
        const line = document.createElement('div');
        line.className = 'log-line';
        line.textContent = e.data;
        logConsole.appendChild(line);
      }
    }
  };
  evtSource.onerror = (err) => {
    console.warn('SSE notification:', err);
  };
} catch (e) {
  console.warn('EventSource setup error:', e);
}

// ── Init & auto-refresh ────────────────────────────────────────────────────────
loadPatients();
setInterval(loadPatients, 10000);   // refresh patient cards every 10s
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8000"))
    print("🏥 Missed-Dose Companion Dashboard")
    print(f"   http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
