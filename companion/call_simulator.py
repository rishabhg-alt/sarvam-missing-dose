"""
Simulates a complete voice check-in call using text-based turn simulation.

In production, this is replaced by real telephony (Exotel/Twilio + Sarvam WebSocket).
The simulator:
  - Optionally generates TTS audio for each assistant turn
  - Runs the full structured conversation via the LLM
  - Evaluates and escalates anomalies
"""
import json
import os
import tempfile
from companion import patient_db, conversation, escalation

def play_audio_if_possible(audio_bytes: bytes, turn_idx: int) -> str:
    """Save TTS audio to a temp file and play it on macOS."""
    path = os.path.join(tempfile.gettempdir(), f"sarvam_tts_{turn_idx}.wav")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    try:
        os.system(f"afplay {path} 2>/dev/null")
    except Exception:
        pass
    return path

def simulate_call(patient_id: str, patient_responses: list = None, play_audio: bool = False):
    """
    Simulate a complete check-in call for the given patient.

    Args:
        patient_id: Patient ID from patients.json
        patient_responses: Pre-scripted patient responses (for demo/testing).
        play_audio: If True, plays TTS audio via afplay (macOS only).
    """
    patient_db.load_patients()
    patient = patient_db.get_patient(patient_id)
    if not patient:
        print(f"❌ Patient {patient_id} not found.")
        return None

    lang = patient["language"]
    print(f"\n{'='*60}")
    print(f"📞 OUTBOUND CALL SIMULATION")
    print(f"   Patient  : {patient['name']} ({patient_id})")
    print(f"   Language : {lang}")
    print(f"   Phone    : {patient['phone']}")
    print(f"{'='*60}\n")
    print("🔔 [Dialling...]\n")

    responses = patient_responses or ["Yes, I am taking my medicines and I feel okay."] * 5

    assistant_replies, report = conversation.run_text_conversation(patient, responses)

    for i, reply in enumerate(assistant_replies):
        # Strip the JSON block from display
        display_reply = reply.split("```json")[0].strip()
        if display_reply:
            print(f"🤖 Agent ({lang}):\n{display_reply}\n")

        if play_audio and display_reply:
            try:
                from companion import tts
                audio = tts.speak(display_reply, lang)
                saved = play_audio_if_possible(audio, i)
                print(f"   🔊 Audio saved: {saved}")
            except Exception as e:
                print(f"   ⚠️  TTS error: {e}")

        if patient_responses and i < len(patient_responses):
            print(f"👤 Patient: {patient_responses[i]}\n")

    print("\n" + "="*60)
    print("📋 CHECK-IN REPORT")
    print("="*60)
    if report:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("⚠️  No structured report could be extracted from the conversation.")

    if report:
        patient_db.record_checkin(patient["id"], report)
        escalation.evaluate_and_escalate(patient, report)

    return report
