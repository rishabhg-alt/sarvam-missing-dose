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
<title>Missed-Dose & Follow-Up Companion · Clinical Voice AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-page: #0b0f19;
    --bg-card: #131b2e;
    --bg-card-hover: #18223a;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-highlight: rgba(99, 102, 241, 0.3);
    --primary: #6366f1;
    --primary-light: #818cf8;
    --primary-glow: rgba(99, 102, 241, 0.25);
    --emerald: #10b981;
    --emerald-light: #34d399;
    --emerald-glow: rgba(16, 185, 129, 0.2);
    --rose: #f43f5e;
    --rose-glow: rgba(244, 63, 94, 0.2);
    --amber: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-page);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.5;
    background-image: 
      radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
      radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
    background-attachment: fixed;
  }

  /* ── Header ── */
  header {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border-color);
    padding: 16px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .brand-group { display: flex; align-items: center; gap: 14px; }
  .brand-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 4px 14px var(--primary-glow);
  }
  header h1 {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .brand-subtitle {
    font-size: 0.78rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 1px;
  }
  .model-pill {
    background: rgba(99, 102, 241, 0.15);
    color: var(--primary-light);
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 1px 7px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
  }
  .header-status {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: var(--emerald-light);
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 0.76rem;
    font-weight: 600;
  }
  .status-dot {
    width: 8px; height: 8px;
    background: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--emerald);
    animation: pulse-dot 2s infinite;
  }
  @keyframes pulse-dot {
    0% { transform: scale(0.9); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.8; }
  }

  /* ── Navigation ── */
  nav {
    display: flex;
    gap: 8px;
    background: rgba(15, 23, 42, 0.4);
    border-bottom: 1px solid var(--border-color);
    padding: 8px 36px;
  }
  nav button {
    padding: 10px 18px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.86rem;
    font-weight: 600;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  nav button:hover {
    color: white;
    background: rgba(255, 255, 255, 0.05);
  }
  nav button.active {
    color: white;
    background: var(--primary);
    box-shadow: 0 2px 10px var(--primary-glow);
  }

  main {
    padding: 32px 36px;
    max-width: 1280px;
    margin: 0 auto;
  }
  section { display: none; }
  section.active { display: block; animation: fadein 0.25s ease-out; }

  @keyframes fadein {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* ── Stats Overview ── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
  }
  .stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
  }
  .stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.15);
  }
  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }
  .stat-card.blue::before { background: linear-gradient(90deg, #6366f1, #3b82f6); }
  .stat-card.green::before { background: linear-gradient(90deg, #10b981, #059669); }
  .stat-card.red::before { background: linear-gradient(90deg, #f43f5e, #e11d48); }
  .stat-card.purple::before { background: linear-gradient(90deg, #a855f7, #6366f1); }
  .stat-card .num {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
  }
  .stat-card.blue .num { color: #818cf8; }
  .stat-card.green .num { color: #34d399; }
  .stat-card.red .num { color: #fb7185; }
  .stat-card.purple .num { color: #c084fc; }
  .stat-card .lbl {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* ── Patient Grid ── */
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  .section-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  .patients-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 20px;
  }
  .patient-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
  }
  .patient-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-highlight);
    box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
    transform: translateY(-2px);
  }
  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
  }
  .patient-identity {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .patient-avatar {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1.05rem;
    color: white;
  }
  .patient-avatar.male {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
  }
  .patient-avatar.female {
    background: linear-gradient(135deg, #ec4899, #be185d);
    box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3);
  }
  .patient-name-block .name {
    font-weight: 700;
    font-size: 1.05rem;
    color: #f1f5f9;
  }
  .patient-name-block .meta {
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
  .chip {
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .chip.lang { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
  .chip.ok { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
  .chip.alert { background: rgba(244, 63, 94, 0.15); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.3); }
  .chip.running { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }

  .meds-container {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 14px;
    border: 1px solid rgba(255, 255, 255, 0.04);
  }
  .meds-header {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 6px;
  }
  .med-pill-item {
    font-size: 0.8rem;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }
  .med-pill-item:last-child { margin-bottom: 0; }
  .appointment-note {
    font-size: 0.76rem;
    color: var(--text-secondary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .card-footer {
    display: flex;
    gap: 10px;
    margin-top: auto;
  }
  .btn {
    padding: 9px 18px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    font-size: 0.84rem;
    font-weight: 600;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    text-decoration: none;
  }
  .btn-primary {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    box-shadow: 0 4px 12px var(--primary-glow);
    flex: 1;
  }
  .btn-primary:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px var(--primary-glow);
  }
  .btn-primary:disabled {
    background: #334155;
    color: #64748b;
    box-shadow: none;
    cursor: not-allowed;
    transform: none;
  }
  .btn-secondary {
    background: rgba(255, 255, 255, 0.06);
    color: #cbd5e1;
    border: 1px solid var(--border-color);
  }
  .btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
    color: white;
  }
  .btn-sm { padding: 6px 12px; font-size: 0.78rem; border-radius: 8px; }

  /* ── Live Call Session Hub ── */
  .call-hub {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.6);
  }
  .hub-header {
    background: rgba(15, 23, 42, 0.8);
    border-bottom: 1px solid var(--border-color);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .caller-profile {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .caller-avatar {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 0 16px var(--primary-glow);
    transition: all 0.3s ease;
  }
  .caller-avatar.ringing {
    animation: ring-pulse 1.4s infinite ease-in-out;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
  }
  @keyframes ring-pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.6); }
    50% { transform: scale(1.08); box-shadow: 0 0 0 12px rgba(245, 158, 11, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
  }
  .caller-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: white;
  }
  .caller-details {
    font-size: 0.78rem;
    color: var(--text-secondary);
  }

  .soundwave-box {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 24px;
  }
  .sound-bar {
    width: 3px;
    background: var(--primary-light);
    border-radius: 3px;
    animation: bounce 1.2s infinite ease-in-out;
  }
  .sound-bar:nth-child(1) { height: 6px; animation-delay: 0.1s; }
  .sound-bar:nth-child(2) { height: 16px; animation-delay: 0.2s; }
  .sound-bar:nth-child(3) { height: 22px; animation-delay: 0.3s; }
  .sound-bar:nth-child(4) { height: 12px; animation-delay: 0.4s; }
  .sound-bar:nth-child(5) { height: 8px; animation-delay: 0.5s; }
  @keyframes bounce {
    0%, 100% { transform: scaleY(0.4); opacity: 0.5; }
    50% { transform: scaleY(1.2); opacity: 1; }
  }

  /* ── Interactive Speech Bubbles ── */
  .hub-body {
    padding: 24px;
    min-height: 280px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: radial-gradient(circle at 50% 10%, rgba(99, 102, 241, 0.04), transparent 60%);
  }
  .bubble-wrap {
    display: flex;
    flex-direction: column;
    max-width: 80%;
    animation: fadein 0.25s ease-out;
  }
  .bubble-wrap.agent { align-self: flex-start; }
  .bubble-wrap.patient { align-self: flex-end; }

  .bubble-speaker-tag {
    font-size: 0.74rem;
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .bubble-wrap.agent .bubble-speaker-tag { color: var(--primary-light); }
  .bubble-wrap.patient .bubble-speaker-tag { color: var(--emerald-light); justify-content: flex-end; }

  .bubble-bubble {
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.94rem;
    line-height: 1.6;
    position: relative;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
  .bubble-wrap.agent .bubble-bubble {
    background: #1e293b;
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #f1f5f9;
    border-bottom-left-radius: 4px;
  }
  .bubble-wrap.patient .bubble-bubble {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #e2e8f0;
    border-bottom-right-radius: 4px;
  }

  .bubble-action-bar {
    margin-top: 6px;
    display: flex;
    gap: 8px;
  }
  .bubble-wrap.patient .bubble-action-bar { justify-content: flex-end; }
  .btn-replay-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .btn-replay-pill:hover {
    background: rgba(255, 255, 255, 0.12);
    color: white;
  }

  .typing-cursor {
    display: inline-block;
    color: var(--primary-light);
    font-weight: 800;
    margin-left: 3px;
    animation: blink 0.7s infinite;
  }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  /* ── Call Conclusion Banner ── */
  .conclusion-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 16px;
    animation: fadein 0.35s ease-out;
  }
  .conclusion-banner.ok {
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
  }
  .conclusion-banner.alert {
    border: 1px solid rgba(244, 63, 94, 0.4);
    box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15);
  }
  .conclusion-title {
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .conclusion-banner.ok .conclusion-title { color: var(--emerald-light); }
  .conclusion-banner.alert .conclusion-title { color: #fb7185; }
  .conclusion-body {
    font-size: 0.88rem;
    color: #cbd5e1;
    margin-bottom: 14px;
    line-height: 1.6;
  }

  /* ── Dialing & Connecting Screen ── */
  .dialing-banner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 50px 24px;
    text-align: center;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px dashed rgba(245, 158, 11, 0.45);
    border-radius: 20px;
    margin: 8px 0;
    animation: fadein 0.3s ease-out;
  }
  .dialing-pulse-ring {
    position: relative;
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: rgba(245, 158, 11, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 18px;
    animation: ring-pulse 1.5s infinite;
  }
  .dialing-pulse-ring .pulse-icon {
    font-size: 2rem;
    animation: shake 1.2s infinite ease-in-out;
  }
  @keyframes shake {
    0%, 100% { transform: rotate(0deg); }
    20%, 60% { transform: rotate(-10deg); }
    40%, 80% { transform: rotate(10deg); }
  }
  .dialing-heading {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
  }
  .dialing-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-bottom: 16px;
    max-width: 440px;
    line-height: 1.5;
  }
  .dialing-meta {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fcd34d;
    font-family: 'JetBrains Mono', monospace;
  }
  .pulsing-amber-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f59e0b;
    box-shadow: 0 0 10px #f59e0b;
    animation: blink 0.8s infinite;
  }

  /* ── Escalations Table ── */
  .table-box {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    overflow: hidden;
  }
  table { width: 100%; border-collapse: collapse; }
  th {
    background: rgba(15, 23, 42, 0.8);
    color: var(--text-muted);
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 14px 20px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
  }
  td {
    padding: 16px 20px;
    font-size: 0.86rem;
    border-bottom: 1px solid var(--border-color);
    color: #cbd5e1;
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255, 255, 255, 0.02); }

  /* ── Modal ── */
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(8px);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: var(--bg-card);
    border: 1px solid var(--border-highlight);
    border-radius: 20px;
    padding: 28px;
    max-width: 620px;
    width: 90%;
    max-height: 85vh;
    overflow-y: auto;
    position: relative;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
  }
  .modal h3 { font-size: 1.2rem; font-weight: 800; margin-bottom: 16px; color: white; }
  .modal-close {
    position: absolute; top: 20px; right: 20px;
    background: rgba(255, 255, 255, 0.08);
    border: none; color: #94a3b8; width: 32px; height: 32px;
    border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .modal-close:hover { background: rgba(255, 255, 255, 0.15); color: white; }

  /* ── Tech Console ── */
  #log-console {
    background: #090d16;
    border: 1px solid var(--border-color);
    color: #34d399;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    padding: 16px;
    border-radius: 12px;
    height: 160px;
    overflow-y: auto;
    line-height: 1.7;
  }

  .empty-state {
    padding: 60px 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.9rem;
  }

  /* ── Floating Jump to Bottom Pill ── */
  .scroll-bottom-pill {
    position: fixed;
    bottom: 24px;
    right: 28px;
    z-index: 99;
    display: none;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 9999px;
    font-family: inherit;
    font-size: 0.85rem;
    font-weight: 700;
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.5);
    cursor: pointer;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
    backdrop-filter: blur(8px);
  }
  .scroll-bottom-pill:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 14px 30px rgba(79, 70, 229, 0.65);
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
  }
  .scroll-bottom-pill:active {
    transform: translateY(0) scale(0.98);
  }
</style>
</head>
<body>

<header>
  <div class="brand-group">
    <div class="brand-icon">🏥</div>
    <div>
      <h1>Missed-Dose & Follow-Up Companion</h1>
      <div class="brand-subtitle">
        AI Post-Discharge Rural Health Agent
        <span class="model-pill">Sarvam 105B</span>
        <span class="model-pill">Bulbul v3 TTS</span>
      </div>
    </div>
  </div>
  <div class="header-status">
    <div class="status-dot"></div>
    <span>Voice Engine Online</span>
  </div>
</header>

<nav>
  <button id="nav-patients" class="active" onclick="showTab('patients')">
    <span>👥</span> Patient Roster
  </button>
  <button id="nav-logs" onclick="showTab('logs')">
    <span>📞</span> Live Call Console
  </button>
  <button id="nav-escalations" onclick="showTab('escalations')">
    <span>🚨</span> Clinical Escalations
  </button>
</nav>

<main>
  <!-- Patients tab -->
  <section id="tab-patients" class="active">
    <div class="stats-grid" id="stats-row"></div>
    <div class="section-header">
      <h2 class="section-title">Enrolled Post-Discharge Patients</h2>
    </div>
    <div class="patients-grid" id="patient-grid">
      <div class="empty-state">Loading clinical records…</div>
    </div>
  </section>

  <!-- Live Call Console tab -->
  <section id="tab-logs">
    <div class="call-hub">
      <div class="hub-header">
        <div class="caller-profile">
          <div class="caller-avatar" id="call-avatar">📞</div>
          <div>
            <div class="caller-name" id="call-patient-name">Ready for Outbound Check-In</div>
            <div class="caller-details" id="call-details">Select any patient from the roster to initiate an AI call in their native language</div>
          </div>
        </div>

        <div style="display:flex; align-items:center; gap:12px">
          <div class="soundwave-box" id="soundwave" style="display:none">
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
          </div>

          <button id="btn-audio-toggle" class="btn btn-sm" style="background:rgba(99, 102, 241, 0.2); color:#a5b4fc; border:1px solid rgba(99, 102, 241, 0.4)" onclick="toggleAudio()">
            🔊 Voice Audio: ACTIVE
          </button>
          <button class="btn btn-sm btn-secondary" onclick="clearLiveCall()">
            🗑 Clear
          </button>
        </div>
      </div>

      <!-- Persistent DOM Audio Channels -->
      <audio id="call-audio" preload="auto" style="display:none"></audio>
      <audio id="replay-audio" preload="auto" style="display:none"></audio>

      <!-- Live Speech Stream -->
      <div class="hub-body" id="call-transcript">
        <div class="empty-state" id="transcript-empty">
          Click <strong>📞 Start Native Call</strong> on any patient to connect live.<br>
          The agent will verify medication doses, screen for worsening symptoms, and confirm clinical follow-ups.
        </div>
      </div>
    </div>

    <!-- Collapsible Diagnostics Console -->
    <details style="background:var(--bg-card); border:1px solid var(--border-color); border-radius:14px; padding:12px 18px">
      <summary style="font-size:0.78rem; font-weight:700; color:var(--text-muted); cursor:pointer">Technical & Escalation Telemetry</summary>
      <div id="log-console" style="margin-top:10px">System standby.</div>
    </details>
  </section>

  <!-- Escalations tab -->
  <section id="tab-escalations">
    <div class="section-header">
      <h2 class="section-title">Clinical Triage & Escalation Register</h2>
    </div>
    <div class="table-box">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th><th>Patient</th><th>Triggers Flagged</th><th>Coordinator Contact</th><th>Outcome Summary</th><th></th>
          </tr>
        </thead>
        <tbody id="escalation-tbody">
          <tr><td colspan="6" class="empty-state">Loading escalation logs…</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</main>

<!-- Detailed Report Modal -->
<div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
  <div class="modal" id="modal-content">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div id="modal-body"></div>
  </div>
</div>

<!-- Floating Scroll to Latest Pill -->
<button id="btn-scroll-bottom" class="scroll-bottom-pill" onclick="scrollToBottomManual()">
  <span>⬇️</span> <span>Jump to Latest</span>
</button>

<script>
const LANG_NAMES = {
  'hi-IN':'Hindi (हिन्दी)','ta-IN':'Tamil (தமிழ்)','te-IN':'Telugu (తెలుగు)','bn-IN':'Bengali (বাংলা)',
  'mr-IN':'Marathi (मराठी)','gu-IN':'Gujarati (ગુજરાતી)','kn-IN':'Kannada (ಕನ್ನಡ)','ml-IN':'Malayalam (മലയാളം)',
  'pa-IN':'Punjabi (ਪੰਜਾਬੀ)','en-IN':'English'
};

let patients = [];
let escalations = [];

// ── Tab switching ────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  
  const targetSection = document.getElementById('tab-' + name);
  if (targetSection) targetSection.classList.add('active');
  
  const targetBtn = document.getElementById('nav-' + name);
  if (targetBtn) targetBtn.classList.add('active');
  
  if (name === 'escalations') loadEscalations();
  else if (name === 'patients') loadPatients();

  updateScrollButton();
}

// ── Patients Roster ──────────────────────────────────────────────────────────
async function loadPatients() {
  const grid = document.getElementById('patient-grid');
  try {
    const res = await fetch('/api/patients');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    patients = await res.json();
    renderStats();
    renderPatients();
  } catch (err) {
    if (grid && !patients.length) {
      grid.innerHTML = '<div class="empty-state" style="color:var(--rose)">Failed to connect to patient registry: ' + err.message + '</div>';
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
      <div class="stat-card blue">
        <div class="num">${total}</div>
        <div class="lbl">Enrolled Patients</div>
      </div>
      <div class="stat-card green">
        <div class="num">${checkedIn}</div>
        <div class="lbl">Check-Ins Completed</div>
      </div>
      <div class="stat-card red">
        <div class="num">${escalated}</div>
        <div class="lbl">Escalations Flagged</div>
      </div>
      <div class="stat-card purple">
        <div class="num">3</div>
        <div class="lbl">Native Languages</div>
      </div>
    `;
  }
}

function renderPatients() {
  const grid = document.getElementById('patient-grid');
  if (!grid) return;
  if (!patients.length) { grid.innerHTML = '<div class="empty-state">No clinical records found.</div>'; return; }
  
  grid.innerHTML = patients.map(p => {
    const lang = LANG_NAMES[p.language] || p.language;
    const lastTime = p.last_checkin ? new Date(p.last_checkin).toLocaleString('en-IN') : 'None yet';
    const isFemale = (p.gender || '').toLowerCase() === 'female';
    const initials = p.name.split(' ').map(n => n[0]).join('').slice(0, 2);
    
    const isThisCalling = (activeCallingPatient && p.id === activeCallingPatient.id && currentCallState !== 'ended' && currentCallState !== 'idle');
    const isRunning = p.job_status === 'running' || isThisCalling;
    
    const statusChip = isRunning
      ? (currentCallState === 'dialing' 
          ? '<span class="chip running">⏳ Dialing…</span>' 
          : '<span class="chip running" style="background:rgba(59,130,246,0.15);color:#93c5fd;border-color:rgba(59,130,246,0.3)">📞 In Call…</span>')
      : p.last_escalated
        ? '<span class="chip alert">🚨 Escalated</span>'
        : p.total_checkins > 0
          ? '<span class="chip ok">✅ Routine Care</span>'
          : '<span class="chip" style="background:rgba(255,255,255,0.06);color:#94a3b8">Pending 1st Check</span>';

    const meds = (p.medications || []).map(m => `<div class="med-pill-item"><span>💊</span> ${m}</div>`).join('');
    const btnText = isRunning
      ? (currentCallState === 'dialing' ? '⏳ Dialing…' : '📞 In Call…')
      : '📞 Start Native Call';
    const btnDisabled = isRunning ? 'disabled' : '';

    return `
    <div class="patient-card" id="card-${p.id}">
      <div>
        <div class="card-top">
          <div class="patient-identity">
            <div class="patient-avatar ${isFemale ? 'female' : 'male'}">${initials}</div>
            <div class="patient-name-block">
              <div class="name">${p.name}</div>
              <div class="meta">${isFemale ? '👩 Female' : '👨 Male'} · ${p.phone}</div>
            </div>
          </div>
          <div>${statusChip}</div>
        </div>

        <div class="badge-row">
          <span class="chip lang">🗣 ${lang}</span>
        </div>

        <div class="meds-container">
          <div class="meds-header">Prescribed Regimen</div>
          ${meds}
        </div>

        <div class="appointment-note">
          <span>📅</span> Next Appointment: <strong>${p.next_appointment}</strong>
        </div>
      </div>

      <div class="card-footer">
        <button class="btn btn-primary" id="btn-${p.id}" onclick="triggerCheckin('${p.id}')" ${btnDisabled}>
          ${btnText}
        </button>
        ${p.total_checkins > 0 ? `<button class="btn btn-secondary" onclick="showHistory('${p.id}')">📋 History</button>` : ''}
      </div>
    </div>`;
  }).join('');
}

// ── Call Progression State ──────────────────────────────────────────────────
let currentCallState = 'idle'; // 'idle' | 'dialing' | 'connecting' | 'speaking' | 'ended'
let activeCallingPatient = null;

async function triggerCheckin(patientId) {
  unlockAudio();
  const patient = patients.find(p => p.id === patientId);
  activeCallingPatient = patient;
  currentCallState = 'dialing';

  const btn = document.getElementById('btn-' + patientId);
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Dialing…';
  }

  showTab('logs');
  autoScrollEnabled = true;
  updateScrollButton();

  // Reset audio & queues for clean call start
  const cAudio = getCallAudio();
  if (cAudio) {
    try { cAudio.pause(); cAudio.currentTime = 0; cAudio.src = ''; } catch (e) {}
  }
  const rAudio = getReplayAudio();
  if (rAudio) {
    try { rAudio.pause(); rAudio.currentTime = 0; rAudio.src = ''; } catch (e) {}
  }
  turnQueue = [];
  isPlayingTurn = false;
  currentTurn = null;
  pendingCallEnd = null;

  const wave = document.getElementById('soundwave');
  if (wave) wave.style.display = 'none';

  if (patient) {
    const pName = document.getElementById('call-patient-name');
    const pDetails = document.getElementById('call-details');
    const pAvatar = document.getElementById('call-avatar');
    if (pName) {
      pName.innerHTML = `<span class="chip running">⏳ DIALING…</span> ${patient.name}`;
    }
    if (pDetails) {
      pDetails.textContent = `Outbound AI Call · ${patient.phone} · ${LANG_NAMES[patient.language] || patient.language}`;
    }
    if (pAvatar) {
      pAvatar.className = 'caller-avatar ringing';
      pAvatar.textContent = '📲';
    }

    const transcript = document.getElementById('call-transcript');
    if (transcript) {
      transcript.innerHTML = `
        <div class="dialing-banner" id="dialing-indicator">
          <div class="dialing-pulse-ring">
            <div class="pulse-icon">📲</div>
          </div>
          <div class="dialing-heading">Dialing ${patient.name}…</div>
          <div class="dialing-sub">Establishing outbound voice channel in <strong>${LANG_NAMES[patient.language] || patient.language}</strong></div>
          <div class="dialing-meta">
            <span class="pulsing-amber-dot"></span> Calling ${patient.phone} · Waiting for connection…
          </div>
        </div>
      `;
    }
  }

  try {
    await fetch('/api/checkin/' + patientId, { method: 'POST' });
  } catch (err) {
    console.error('Trigger checkin failed:', err);
  }

  const poll = setInterval(async () => {
    await loadPatients();
    const p = patients.find(x => x.id === patientId);
    if (p && p.job_status !== 'running') {
      clearInterval(poll);
      if (currentCallState === 'ended' || currentCallState === 'idle') {
        if (btn) {
          btn.disabled = false;
          btn.textContent = '📞 Start Native Call';
        }
      }
    }
  }, 3000);
}

// ── Continuous Unified Audio Engine ──────────────────────────────────────────
let audioEnabled = true;
let audioUnlocked = false;
let isPlayingTurn = false;
let currentTurn = null;
let turnQueue = [];
let pendingCallEnd = null;

// ── Smart Auto-Scroll System ────────────────────────────────────────────────
let autoScrollEnabled = true;
let isProgrammaticScroll = false;
let programmaticScrollTimer = null;

function getDistanceFromBottom() {
  const scrollHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
  const scrollTop = window.scrollY || window.pageYOffset || document.documentElement.scrollTop;
  const clientHeight = window.innerHeight || document.documentElement.clientHeight;
  return scrollHeight - (scrollTop + clientHeight);
}

function updateScrollButton() {
  const btn = document.getElementById('btn-scroll-bottom');
  if (!btn) return;
  const logsTab = document.getElementById('tab-logs');
  const isLogsActive = logsTab && logsTab.classList.contains('active');
  const transcript = document.getElementById('call-transcript');
  const hasContent = transcript && !document.getElementById('transcript-empty');
  const dist = getDistanceFromBottom();
  if (isLogsActive && !autoScrollEnabled && hasContent && dist > 70) {
    btn.style.display = 'inline-flex';
  } else {
    btn.style.display = 'none';
  }
}

function smartScrollToBottom() {
  if (!autoScrollEnabled) return;
  const dist = getDistanceFromBottom();
  if (dist > 15) {
    isProgrammaticScroll = true;
    clearTimeout(programmaticScrollTimer);
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    programmaticScrollTimer = setTimeout(() => {
      isProgrammaticScroll = false;
    }, 350);
  }
}

function scrollToBottomManual() {
  autoScrollEnabled = true;
  updateScrollButton();
  isProgrammaticScroll = true;
  clearTimeout(programmaticScrollTimer);
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  programmaticScrollTimer = setTimeout(() => {
    isProgrammaticScroll = false;
  }, 400);
}

// User-initiated scroll listeners to detect user scrolling above
window.addEventListener('wheel', (e) => {
  if (e.deltaY < 0) {
    // User is scrolling UP
    autoScrollEnabled = false;
    isProgrammaticScroll = false;
    clearTimeout(programmaticScrollTimer);
    updateScrollButton();
  } else if (e.deltaY > 0) {
    // User is scrolling DOWN
    setTimeout(() => {
      if (getDistanceFromBottom() < 60) {
        autoScrollEnabled = true;
        updateScrollButton();
      }
    }, 50);
  }
}, { passive: true });

let touchStartY = 0;
window.addEventListener('touchstart', (e) => {
  if (e.touches && e.touches.length > 0) {
    touchStartY = e.touches[0].clientY;
  }
}, { passive: true });

window.addEventListener('touchmove', (e) => {
  if (e.touches && e.touches.length > 0) {
    const currentY = e.touches[0].clientY;
    if (currentY > touchStartY + 15) {
      // Swiping finger downward scrolls page upward
      autoScrollEnabled = false;
      isProgrammaticScroll = false;
      clearTimeout(programmaticScrollTimer);
      updateScrollButton();
    } else if (currentY < touchStartY - 15) {
      setTimeout(() => {
        if (getDistanceFromBottom() < 60) {
          autoScrollEnabled = true;
          updateScrollButton();
        }
      }, 50);
    }
  }
}, { passive: true });

window.addEventListener('keydown', (e) => {
  if (['PageUp', 'ArrowUp', 'Home'].includes(e.key)) {
    autoScrollEnabled = false;
    isProgrammaticScroll = false;
    clearTimeout(programmaticScrollTimer);
    updateScrollButton();
  } else if (['PageDown', 'ArrowDown', 'End'].includes(e.key)) {
    setTimeout(() => {
      if (getDistanceFromBottom() < 60) {
        autoScrollEnabled = true;
        updateScrollButton();
      }
    }, 80);
  }
});

window.addEventListener('scroll', () => {
  if (isProgrammaticScroll) return;
  const dist = getDistanceFromBottom();
  if (dist > 120) {
    autoScrollEnabled = false;
  } else if (dist < 50) {
    autoScrollEnabled = true;
  }
  updateScrollButton();
}, { passive: true });

function getCallAudio() {
  return document.getElementById('call-audio');
}
function getReplayAudio() {
  return document.getElementById('replay-audio');
}

function updateAudioButton() {
  const btn = document.getElementById('btn-audio-toggle');
  if (btn) {
    btn.innerHTML = audioEnabled ? '🔊 Voice Audio: ACTIVE' : '🔈 Voice Audio: MUTED';
    btn.style.background = audioEnabled ? 'rgba(99, 102, 241, 0.2)' : 'rgba(244, 63, 94, 0.2)';
    btn.style.color = audioEnabled ? '#a5b4fc' : '#fda4af';
    btn.style.borderColor = audioEnabled ? 'rgba(99, 102, 241, 0.4)' : 'rgba(244, 63, 94, 0.4)';
  }
}

function unlockAudio() {
  if (!audioUnlocked) {
    try {
      const dummy = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA");
      dummy.play().then(() => {
        audioUnlocked = true;
      }).catch(() => {});
    } catch (e) {}
  }
}

// Seamless Hardware Mute / Unmute (Continuously Playing in Background)
function toggleAudio() {
  unlockAudio();
  audioEnabled = !audioEnabled;
  updateAudioButton();

  const cAudio = getCallAudio();
  const rAudio = getReplayAudio();

  // Volume & muted property toggle happens live without stopping, pausing, or reloading!
  if (cAudio) {
    cAudio.muted = !audioEnabled;
    cAudio.volume = audioEnabled ? 1.0 : 0.0;
  }
  if (rAudio) {
    rAudio.muted = !audioEnabled;
    rAudio.volume = audioEnabled ? 1.0 : 0.0;
  }
}

function enqueueTurn(turn) {
  // Pre-load audio into browser cache
  const audioUrl = '/api/speak?role=' + turn.role + '&lang=' + encodeURIComponent(turn.lang) + '&speaker=' + encodeURIComponent(turn.speaker_voice || '') + '&gender=' + encodeURIComponent(turn.gender || (turn.role === 'agent' ? 'female' : 'male')) + '&text=' + encodeURIComponent(turn.text);
  const pre = new Audio(audioUrl);
  pre.preload = "auto";
  pre.load();

  turnQueue.push(turn);
  if (!isPlayingTurn) {
    playNextTurn();
  }
}

function createSpeechBubble(role, speaker, lang, fullText, speakerVoice, gender) {
  const transcript = document.getElementById('call-transcript');
  if (!transcript) return null;
  const empty = document.getElementById('transcript-empty');
  if (empty) empty.remove();
  const dialingInd = document.getElementById('dialing-indicator');
  if (dialingInd) dialingInd.remove();

  // Transition header from connecting/dialing to active live call
  if (currentCallState !== 'speaking') {
    currentCallState = 'speaking';
    const pName = document.getElementById('call-patient-name');
    const pDetails = document.getElementById('call-details');
    const pAvatar = document.getElementById('call-avatar');
    if (pName && activeCallingPatient) {
      pName.innerHTML = `<span class="chip ok">🟢 LIVE CALL</span> ${activeCallingPatient.name}`;
    }
    if (pDetails && activeCallingPatient) {
      pDetails.textContent = `Active Check-In · ${activeCallingPatient.phone} · ${LANG_NAMES[activeCallingPatient.language] || activeCallingPatient.language}`;
    }
    if (pAvatar && activeCallingPatient) {
      pAvatar.className = 'caller-avatar';
      pAvatar.textContent = (activeCallingPatient.gender === 'female' ? '👩' : '👨');
    }
  }

  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap ' + (role === 'agent' ? 'agent' : 'patient');

  const sender = document.createElement('div');
  sender.className = 'bubble-speaker-tag';
  sender.innerHTML = (role === 'agent' ? '🤖 ' : '👤 ') + speaker;

  const bubble = document.createElement('div');
  bubble.className = 'bubble-bubble';
  bubble.innerHTML = '<span class="typing-cursor">▌</span>';

  const action = document.createElement('div');
  action.className = 'bubble-action-bar';
  action.style.display = 'none';

  const playBtn = document.createElement('button');
  playBtn.className = 'btn-replay-pill';
  playBtn.innerHTML = '🔊 Replay Voice';
  playBtn.onclick = () => playSingleTurn(role, lang, fullText, speakerVoice, gender);
  action.appendChild(playBtn);

  wrap.appendChild(sender);
  wrap.appendChild(bubble);
  wrap.appendChild(action);

  transcript.appendChild(wrap);
  smartScrollToBottom();

  return { wrap, bubble, action };
}

function playNextTurn() {
  if (!turnQueue.length) {
    isPlayingTurn = false;
    currentTurn = null;
    const wave = document.getElementById('soundwave');
    if (wave) wave.style.display = 'none';

    // When all audio turns have concluded, reveal conclusion report!
    if (pendingCallEnd) {
      applyCallEnd(pendingCallEnd);
      pendingCallEnd = null;
    }
    return;
  }

  isPlayingTurn = true;
  const turn = turnQueue.shift();
  currentTurn = turn;

  const bubbleElements = createSpeechBubble(turn.role, turn.speaker, turn.lang, turn.text, turn.speaker_voice, turn.gender);
  const wave = document.getElementById('soundwave');
  if (wave) wave.style.display = 'flex';

  const cAudio = getCallAudio();
  const audioUrl = '/api/speak?role=' + turn.role + '&lang=' + encodeURIComponent(turn.lang) + '&speaker=' + encodeURIComponent(turn.speaker_voice || '') + '&gender=' + encodeURIComponent(turn.gender || (turn.role === 'agent' ? 'female' : 'male')) + '&text=' + encodeURIComponent(turn.text);

  cAudio.onplaying = null;
  cAudio.ontimeupdate = null;
  cAudio.onended = null;
  cAudio.onerror = null;

  cAudio.muted = !audioEnabled;
  cAudio.volume = audioEnabled ? 1.0 : 0.0;
  cAudio.src = audioUrl;

  const words = turn.text.split(' ');
  let turnFinished = false;

  function finishTurn() {
    if (turnFinished) return;
    turnFinished = true;
    if (bubbleElements) {
      if (bubbleElements.bubble) bubbleElements.bubble.textContent = turn.text;
      if (bubbleElements.action) bubbleElements.action.style.display = 'flex';
    }
    currentTurn = null;
    // Small natural pause before other party speaks
    setTimeout(playNextTurn, 120);
  }

  // Audio-Driven Real-Time Word Sync (Single Source of Truth)
  cAudio.ontimeupdate = () => {
    if (turnFinished || !cAudio.duration) return;
    const progress = Math.min(1.0, cAudio.currentTime / cAudio.duration);
    const wordsToShow = Math.max(1, Math.ceil(progress * words.length));
    const visibleText = words.slice(0, wordsToShow).join(' ');

    if (bubbleElements && bubbleElements.bubble) {
      bubbleElements.bubble.textContent = visibleText;
      if (wordsToShow < words.length) {
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        cursor.textContent = ' ▌';
        bubbleElements.bubble.appendChild(cursor);
      }
    }
    smartScrollToBottom();
  };

  cAudio.onended = () => {
    finishTurn();
  };

  cAudio.onerror = (e) => {
    console.warn('Audio stream error, displaying text visually:', e);
    if (bubbleElements && bubbleElements.bubble) {
      bubbleElements.bubble.textContent = turn.text;
    }
    setTimeout(finishTurn, (words.length * 200) + 200);
  };

  cAudio.play().catch(e => {
    console.warn('Audio play request status:', e);
  });
}

function playSingleTurn(role, lang, text, speakerVoice, gender) {
  unlockAudio();
  const rAudio = getReplayAudio();
  if (!rAudio) return;

  if (!audioEnabled) {
    audioEnabled = true;
    updateAudioButton();
    const cAudio = getCallAudio();
    if (cAudio) {
      cAudio.muted = false;
      cAudio.volume = 1.0;
    }
  }

  const audioUrl = '/api/speak?role=' + role + '&lang=' + encodeURIComponent(lang) + '&speaker=' + encodeURIComponent(speakerVoice || '') + '&gender=' + encodeURIComponent(gender || (role === 'agent' ? 'female' : 'male')) + '&text=' + encodeURIComponent(text);
  rAudio.muted = !audioEnabled;
  rAudio.volume = audioEnabled ? 1.0 : 0.0;
  rAudio.src = audioUrl;
  rAudio.play().catch(e => console.warn('Replay audio error:', e));
}

function applyCallEnd(data) {
  currentCallState = 'ended';
  activeCallingPatient = null;
  const pAvatar = document.getElementById('call-avatar');
  if (pAvatar) {
    pAvatar.className = 'caller-avatar';
  }

  const pName = document.getElementById('call-patient-name');
  if (pName) {
    pName.innerHTML = data.escalated 
      ? '<span style="color:#fb7185">Call Ended · Escalation Triggered 🚨</span>' 
      : '<span style="color:#34d399">Call Ended · Patient Adherent ✅</span>';
  }

  const transcript = document.getElementById('call-transcript');
  if (transcript && !document.getElementById('call-conclusion-card')) {
    const card = document.createElement('div');
    card.id = 'call-conclusion-card';
    card.className = 'conclusion-banner ' + (data.escalated ? 'alert' : 'ok');

    card.innerHTML = `
      <div class="conclusion-title">
        <span>${data.escalated ? '🚨' : '✅'}</span>
        <span>${data.escalated ? 'Clinical Escalation Required' : 'Check-In Completed — Routine Care'}</span>
      </div>
      <div class="conclusion-body">
        ${data.report ? (data.report.summary || 'Post-discharge check-in completed.') : 'Check-in completed.'}
      </div>
      <button class="btn btn-sm btn-primary" onclick="showHistory('${data.patient_id}')">
        📋 View Clinical Report & Details
      </button>
    `;
    transcript.appendChild(card);
    smartScrollToBottom();
  }

  loadPatients();
}

function clearLiveCall() {
  currentCallState = 'idle';
  activeCallingPatient = null;
  autoScrollEnabled = true;
  updateScrollButton();

  const cAudio = getCallAudio();
  if (cAudio) {
    try { cAudio.pause(); cAudio.currentTime = 0; cAudio.src = ''; } catch (e) {}
  }
  const rAudio = getReplayAudio();
  if (rAudio) {
    try { rAudio.pause(); rAudio.currentTime = 0; rAudio.src = ''; } catch (e) {}
  }
  const transcript = document.getElementById('call-transcript');
  if (transcript) {
    transcript.innerHTML = `
      <div class="empty-state" id="transcript-empty">
        Click <strong>📞 Start Native Call</strong> on any patient to connect live.<br>
        The agent will verify medication doses, screen for worsening symptoms, and confirm clinical follow-ups.
      </div>
    `;
  }
  const wave = document.getElementById('soundwave');
  if (wave) wave.style.display = 'none';
  const pName = document.getElementById('call-patient-name');
  if (pName) pName.textContent = 'Ready for Outbound Check-In';
  const pDetails = document.getElementById('call-details');
  if (pDetails) pDetails.textContent = 'Select any patient from the roster to initiate an AI call in their native language';
  const pAvatar = document.getElementById('call-avatar');
  if (pAvatar) pAvatar.textContent = '📞';

  turnQueue = [];
  isPlayingTurn = false;
  currentTurn = null;
  pendingCallEnd = null;
}

// ── Heartbeat (Render Free-Tier 24/7) ──────────────────────────────────────────
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
    if (!history.length) { alert('No check-in history recorded yet.'); return; }
    renderModal(patient, history[0]);
  } catch (err) {
    alert('Failed to load history: ' + err.message);
  }
}

function renderModal(patient, checkin) {
  const r = checkin.report || {};
  const meds = Object.entries(r.medication_adherence || {}).map(([med, status]) =>
    `<div style="display:flex; justify-content:space-between; padding:8px 12px; background:rgba(15,23,42,0.6); border-radius:8px; margin-bottom:6px; font-size:0.86rem">
      <span>💊 ${med}</span>
      <span class="chip ${status === 'taken' ? 'ok' : 'alert'}">${status.toUpperCase()}</span>
    </div>`
  ).join('');

  const symptoms = (r.symptoms_reported || []).length
    ? r.symptoms_reported.map(s => `<span class="chip alert">⚠️ ${s}</span>`).join(' ')
    : '<span style="color:#64748b; font-size:0.85rem">None reported</span>';

  const flags = Object.entries(r.escalation_flags || {})
    .filter(([,v]) => v)
    .map(([k]) => `<span class="chip alert">🚨 ${k.replace(/_/g,' ')}</span>`)
    .join(' ') || '<span class="chip ok">✅ None</span>';

  document.getElementById('modal-body').innerHTML = `
    <h3>📋 Check-In Clinical Report</h3>
    <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:18px">
      Patient: <strong style="color:white">${patient ? patient.name : 'Unknown'}</strong> · ${new Date(checkin.timestamp).toLocaleString('en-IN')}
    </div>
    <div style="margin-bottom:18px">
      <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px">Medication Adherence</div>
      ${meds || '<span style="color:#64748b; font-size:0.85rem">None recorded</span>'}
    </div>
    <div style="margin-bottom:18px">
      <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px">Reported Symptoms</div>
      <div>${symptoms}</div>
    </div>
    <div style="margin-bottom:18px">
      <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px">Appointment Confirmed</div>
      <span class="chip ${r.appointment_confirmed ? 'ok' : 'alert'}">${r.appointment_confirmed ? 'Yes, Confirmed' : 'No / Unconfirmed'}</span>
    </div>
    <div style="margin-bottom:18px">
      <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px">Escalation Triggers</div>
      <div>${flags}</div>
    </div>
    <div>
      <div style="font-size:0.75rem; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px">Clinical Summary</div>
      <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25); border-radius:10px; padding:12px; font-size:0.88rem; color:#cbd5e1; line-height:1.6">
        ${r.summary || '—'}
      </div>
    </div>
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
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No escalations yet — all patients adherent! 🎉</td></tr>';
      return;
    }
    tbody.innerHTML = escalations.map((e, index) => {
      const triggers = (Array.isArray(e.triggers) ? e.triggers : JSON.parse(e.triggers || '[]'))
        .map(t => `<span class="chip alert">${t.replace(/_/g,' ')}</span>`).join(' ');
      const report = typeof e.report === 'string' ? JSON.parse(e.report) : (e.report || {});
      return `<tr>
        <td style="white-space:nowrap; color:#94a3b8">${new Date(e.timestamp).toLocaleString('en-IN')}</td>
        <td><strong style="color:white">${e.patient_name}</strong><br><span style="font-size:0.75rem; color:#64748b">${e.patient_id}</span></td>
        <td>${triggers}</td>
        <td style="font-size:0.8rem">${e.coordinator_email}<br>${e.coordinator_phone}</td>
        <td style="font-size:0.82rem; color:#cbd5e1; max-width:240px">${report.summary || '—'}</td>
        <td><button class="btn btn-sm btn-secondary" onclick="showEscalationDetail(${index})">View</button></td>
      </tr>`;
    }).join('');
  } catch (err) {
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state" style="color:var(--rose)">Failed to load escalations: ' + err.message + '</td></tr>';
    }
  }
}

function showEscalationDetail(index) {
  const e = escalations[index];
  if (!e) return;
  const report = typeof e.report === 'string' ? JSON.parse(e.report) : (e.report || {});
  renderModal({ id: e.patient_id, name: e.patient_name }, { report, timestamp: e.timestamp });
}

// ── SSE Live Pipeline ────────────────────────────────────────────────────────
const logConsole = document.getElementById('log-console');
try {
  const evtSource = new EventSource('/api/logs');
  evtSource.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data);
      if (typeof parsed === 'object' && parsed !== null) {
        if (parsed.type === 'call_start') {
          currentCallState = 'connecting';
          const pName = document.getElementById('call-patient-name');
          const pDetails = document.getElementById('call-details');
          const pAvatar = document.getElementById('call-avatar');
          if (pName) {
            pName.innerHTML = `<span class="chip running" style="background:rgba(59,130,246,0.15);color:#93c5fd;border-color:rgba(59,130,246,0.3)">📞 CONNECTING…</span> ${parsed.patient_name}`;
          }
          if (pDetails) {
            pDetails.textContent = `Call Connected · Initializing Sarvam AI Follow-Up Companion (${LANG_NAMES[parsed.language] || parsed.language})…`;
          }
          if (pAvatar) {
            pAvatar.className = 'caller-avatar ringing';
            pAvatar.textContent = '📲';
          }
          const ind = document.getElementById('dialing-indicator');
          if (ind) {
            const h = ind.querySelector('.dialing-heading');
            if (h) h.textContent = `Connecting with ${parsed.patient_name}…`;
            const s = ind.querySelector('.dialing-sub');
            if (s) s.textContent = `Call answered. Synthesizing native speech in ${LANG_NAMES[parsed.language] || parsed.language}…`;
            const m = ind.querySelector('.dialing-meta');
            if (m) m.innerHTML = `<span class="pulsing-amber-dot" style="background:#3b82f6;box-shadow:0 0 10px #3b82f6"></span> Preparing opening healthcare check-in…`;
          }
        } else if (parsed.type === 'turn') {
          enqueueTurn(parsed);
        } else if (parsed.type === 'call_end') {
          pendingCallEnd = parsed;
          if (!isPlayingTurn && !turnQueue.length) {
            applyCallEnd(pendingCallEnd);
            pendingCallEnd = null;
          }
        } else if (parsed.type === 'log') {
          if (logConsole) {
            const line = document.createElement('div');
            line.textContent = parsed.text;
            logConsole.appendChild(line);
            logConsole.scrollTop = logConsole.scrollHeight;
          }
        }
      }
    } catch (err) {}
  };
} catch (e) {
  console.warn('EventSource error:', e);
}

// Initial Boot
loadPatients();
setInterval(loadPatients, 12000);
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
