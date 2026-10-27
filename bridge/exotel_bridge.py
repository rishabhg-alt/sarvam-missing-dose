"""
Exotel WebSocket Audio Bridge
==============================
Exotel streams 8kHz mulaw audio to this server over WebSocket.
This server:
  1. Receives audio chunks from Exotel
  2. Accumulates until silence (VAD) or sentence boundary
  3. Sends to Sarvam STT (saaras:v4) → transcript
  4. Sends transcript to Sarvam LLM → response in patient's language
  5. Sends response to Sarvam TTS (bulbul:v4-flash) → audio
  6. Streams audio back to Exotel

Start with:
    pip install fastapi uvicorn websockets
    python bridge/exotel_bridge.py

Exotel webhook URL → https://<your-ngrok>.ngrok.io/call/connect
Exotel stream URL  → wss://<your-ngrok>.ngrok.io/call/stream/{call_sid}

Environment variables required:
  SARVAM_API_KEY       — your sk_... key
  EXOTEL_SID           — your Exotel account SID
  EXOTEL_API_KEY       — Exotel API key
  EXOTEL_API_TOKEN     — Exotel API token
  EXOTEL_PHONE_NUMBER  — your Exotel virtual number (ExoPhone)
"""
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

# Add parent dir to path so we can import companion modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from companion import patient_db, conversation
from companion.config import VOICE_MAP, LANGUAGE_NAMES

app = FastAPI(title="Missed-Dose Voice Bridge")

# ── Sarvam endpoints ────────────────────────────────────────────────────────
SARVAM_API_KEY = os.environ["SARVAM_API_KEY"]
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# ── In-memory call state ─────────────────────────────────────────────────────
active_calls: dict[str, dict] = {}


def get_patient_for_call(call_sid: str) -> dict | None:
    """Retrieve patient context for a call. In production, map call_sid → patient_id via your DB."""
    return active_calls.get(call_sid, {}).get("patient")


def build_conversation_state(patient: dict) -> dict:
    from companion.conversation import build_system_prompt
    return {
        "patient": patient,
        "messages": [
            {"role": "system", "content": build_system_prompt(patient)},
            {"role": "user", "content": f"[CALL CONNECTED] Patient {patient['name']} has answered. Begin the check-in."}
        ],
        "audio_buffer": b"",
        "report": None,
    }


# ── Sarvam STT (REST) ─────────────────────────────────────────────────────────
async def transcribe_audio(audio_bytes: bytes, language_code: str) -> str:
    """Send audio to Sarvam STT, return transcript."""
    async with httpx.AsyncClient(timeout=15) as client:
        files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
        headers = {"api-subscription-key": SARVAM_API_KEY}
        data = {
            "model": "saaras:v4",
            "language_code": language_code,
            "mode": "transcribe",
        }
        resp = await client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
        resp.raise_for_status()
        return resp.json().get("transcript", "")


# ── Sarvam LLM ────────────────────────────────────────────────────────────────
async def get_llm_reply(messages: list) -> str:
    """Get next agent reply from sarvam-105b-conversations."""
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }
        body = {
            "model": "sarvam-105b-conversations",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 300,
        }
        resp = await client.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Sarvam TTS (REST) ─────────────────────────────────────────────────────────
async def synthesize_speech(text: str, language_code: str) -> bytes:
    """Convert text to 8kHz mulaw audio for telephony."""
    speaker = VOICE_MAP.get(language_code, "ritu")
    async with httpx.AsyncClient(timeout=15) as client:
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }
        body = {
            "text": text[:2500],
            "language_code": language_code,
            "speaker": speaker,
            "model": "bulbul:v4-flash",
            "speech_sample_rate": 8000,
            "output_audio_codec": "wav",
        }
        resp = await client.post(SARVAM_TTS_URL, headers=headers, json=body)
        resp.raise_for_status()
        audio_b64 = resp.json()["audios"][0]
        return base64.b64decode(audio_b64)


# ── Exotel WebSocket Handler ──────────────────────────────────────────────────
@app.websocket("/call/stream/{call_sid}")
async def exotel_stream(websocket: WebSocket, call_sid: str):
    """
    WebSocket endpoint Exotel connects to for bidirectional audio streaming.
    Receives mulaw audio frames, responds with agent audio.
    """
    await websocket.accept()
    print(f"[{call_sid}] WebSocket connected")

    patient = get_patient_for_call(call_sid)
    if not patient:
        print(f"[{call_sid}] No patient context — using default English")
        patient = {"id": "UNKNOWN", "name": "Patient", "language": "en-IN",
                   "medications": [], "next_appointment": "TBD",
                   "discharge_date": "2026-01-01"}

    state = build_conversation_state(patient)
    lang = patient["language"]

    # Send opening greeting immediately
    try:
        opening_text = await get_llm_reply(state["messages"])
        state["messages"].append({"role": "assistant", "content": opening_text})
        audio = await synthesize_speech(opening_text.split("```json")[0].strip(), lang)
        await websocket.send_bytes(audio)
        print(f"[{call_sid}] Sent greeting ({len(audio)} bytes)")
    except Exception as e:
        print(f"[{call_sid}] Greeting error: {e}")

    # Accumulate patient audio chunks
    audio_buffer = b""
    SILENCE_THRESHOLD_BYTES = 16000  # ~2s of 8kHz audio before processing

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                audio_buffer += data["bytes"]

                # Process when we have enough audio (simple length-based VAD)
                if len(audio_buffer) >= SILENCE_THRESHOLD_BYTES:
                    print(f"[{call_sid}] Processing {len(audio_buffer)} bytes of audio")
                    try:
                        transcript = await transcribe_audio(audio_buffer, lang)
                        audio_buffer = b""

                        if not transcript.strip():
                            continue

                        print(f"[{call_sid}] Patient said: {transcript}")
                        state["messages"].append({"role": "user", "content": transcript})

                        reply = await get_llm_reply(state["messages"])
                        state["messages"].append({"role": "assistant", "content": reply})

                        # Check for report
                        from companion.conversation import extract_json_report
                        report = extract_json_report(reply)
                        if report:
                            state["report"] = report
                            print(f"[{call_sid}] Report extracted: {json.dumps(report, indent=2)}")

                        display_reply = reply.split("```json")[0].strip()
                        if display_reply:
                            audio = await synthesize_speech(display_reply, lang)
                            await websocket.send_bytes(audio)

                    except Exception as e:
                        print(f"[{call_sid}] STT/LLM/TTS error: {e}")

            elif "text" in data:
                # Exotel may send JSON control frames
                try:
                    frame = json.loads(data["text"])
                    if frame.get("event") == "stop":
                        print(f"[{call_sid}] Call ended by Exotel")
                        break
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[{call_sid}] WebSocket disconnected")
    finally:
        # Save check-in if report was extracted
        if state.get("report"):
            from companion.storage import save_checkin
            from companion.escalation import evaluate_and_escalate
            from companion.notifier import notify
            save_checkin(patient, state["report"])
            triggered = evaluate_and_escalate(patient, state["report"])
            if triggered:
                notify(patient, triggered, state["report"])
        active_calls.pop(call_sid, None)
        print(f"[{call_sid}] Call cleaned up")


# ── Exotel Outbound Call Trigger ──────────────────────────────────────────────
@app.post("/trigger-call/{patient_id}")
async def trigger_call(patient_id: str, request: Request):
    """
    Trigger an outbound call to a patient via Exotel.
    POST /trigger-call/P001

    Requires in .env:
      EXOTEL_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_PHONE_NUMBER
      PUBLIC_URL  (your ngrok/VPS URL, e.g. https://abc123.ngrok.io)
    """
    patient_db.load_patients()
    patient = patient_db.get_patient(patient_id)
    if not patient:
        return JSONResponse({"error": f"Patient {patient_id} not found"}, status_code=404)

    exotel_sid = os.getenv("EXOTEL_SID")
    exotel_key = os.getenv("EXOTEL_API_KEY")
    exotel_token = os.getenv("EXOTEL_API_TOKEN")
    exotel_phone = os.getenv("EXOTEL_PHONE_NUMBER")
    public_url = os.getenv("PUBLIC_URL", "https://your-server.ngrok.io")

    if not all([exotel_sid, exotel_key, exotel_token, exotel_phone]):
        return JSONResponse(
            {"error": "Exotel credentials not configured. Set EXOTEL_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_PHONE_NUMBER in .env"},
            status_code=400
        )

    # Create a temporary call SID placeholder
    import uuid
    call_sid = str(uuid.uuid4())
    active_calls[call_sid] = build_conversation_state(patient)

    # Exotel API call
    exotel_url = f"https://api.exotel.com/v1/Accounts/{exotel_sid}/Calls/connect.json"
    payload = {
        "From": patient["phone"],
        "To": exotel_phone,
        "Sid": exotel_sid,
        "CallType": "trans",
        "Url": f"{public_url}/call/connect",
        "StatusCallback": f"{public_url}/call/status",
        "CustomField": json.dumps({"patient_id": patient_id, "call_sid": call_sid}),
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                exotel_url,
                data=payload,
                auth=(exotel_key, exotel_token),
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()
            exotel_call_sid = result.get("Call", {}).get("Sid", call_sid)
            active_calls[exotel_call_sid] = active_calls.pop(call_sid, {})
            print(f"[OUTBOUND] Call initiated for {patient['name']}: {exotel_call_sid}")
            return JSONResponse({"call_sid": exotel_call_sid, "patient": patient["name"], "phone": patient["phone"]})
        except Exception as e:
            active_calls.pop(call_sid, None)
            return JSONResponse({"error": str(e)}, status_code=500)


# ── Exotel Connect Webhook ─────────────────────────────────────────────────────
@app.post("/call/connect")
async def call_connect(request: Request):
    """
    Exotel calls this webhook when call is answered.
    Returns TwiML-like XML to instruct Exotel to stream audio to our WebSocket.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    custom = json.loads(form.get("CustomField", "{}"))
    patient_id = custom.get("patient_id")
    public_url = os.getenv("PUBLIC_URL", "https://your-server.ngrok.io")

    if patient_id:
        patient_db.load_patients()
        patient = patient_db.get_patient(patient_id)
        if patient and call_sid not in active_calls:
            active_calls[call_sid] = build_conversation_state(patient)

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{public_url.replace('https://', '').replace('http://', '')}/call/stream/{call_sid}">
            <Parameter name="patient_id" value="{patient_id or ''}"/>
        </Stream>
    </Connect>
</Response>"""
    return PlainTextResponse(xml, media_type="application/xml")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "active_calls": len(active_calls)}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    patient_db.load_patients()
    from companion.storage import init_db
    init_db()
    print("🚀 Exotel Bridge running on http://0.0.0.0:8000")
    print("   WebSocket endpoint: ws://localhost:8000/call/stream/{call_sid}")
    print("   Trigger call:       POST /trigger-call/{patient_id}")
    print("   Health:             GET  /health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
