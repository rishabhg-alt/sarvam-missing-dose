"""
LLM-driven structured conversation engine for patient check-ins.

Flow:
  1. Greet patient in their language
  2. Ask about each medication
  3. Ask about symptoms
  4. Confirm upcoming appointment
  5. Produce structured JSON check-in report
"""
import json
import os
import re
from datetime import date
from dotenv import load_dotenv
from sarvamai import SarvamAI
from companion.config import CHAT_MODEL, LANGUAGE_NAMES

load_dotenv()
_client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

SYSTEM_PROMPT = """You are a warm, empathetic healthcare follow-up assistant calling post-discharge rural patients on behalf of their clinic.

Your role:
- Speak ONLY in the patient's specified language: {language_name}
- Keep sentences SHORT and simple (rural patients may have low health literacy)
- Be warm and reassuring, never clinical or alarming
- Conduct a STRUCTURED check-in covering exactly these 3 areas:
  1. MEDICATION: Ask about each medication by name. Did they take it today? Any difficulties?
  2. SYMPTOMS: Ask if they have any new or worsening symptoms since discharge
  3. APPOINTMENT: Remind them of their next appointment ({next_appointment}) and confirm they plan to attend

After completing all 3 areas, output a structured JSON summary (always in English) wrapped in triple backtick json markers with this exact schema:
{{
  "medication_adherence": {{
    "<medication_name>": "taken" | "missed" | "partial" | "unknown"
  }},
  "symptoms_reported": ["list of symptoms mentioned, empty list if none"],
  "appointment_confirmed": true,
  "escalation_flags": {{
    "missed_critical_dose": false,
    "severe_symptom_reported": false,
    "missed_appointment": false,
    "patient_confused": false,
    "emergency_symptom": false
  }},
  "summary": "One English sentence summarizing the call outcome"
}}

Patient info:
- Name: {name}
- Medications: {medications}
- Next appointment: {next_appointment}
- Days since discharge: {days_since_discharge}
"""

def build_system_prompt(patient: dict) -> str:
    discharge = date.fromisoformat(patient["discharge_date"])
    days = (date.today() - discharge).days
    lang = patient["language"]
    return SYSTEM_PROMPT.format(
        language_name=LANGUAGE_NAMES.get(lang, lang),
        name=patient["name"],
        medications=", ".join(patient["medications"]),
        next_appointment=patient["next_appointment"],
        days_since_discharge=days,
    )

def extract_json_report(text: str):
    """Extract structured JSON block from LLM response."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None

def run_text_conversation(patient: dict, user_inputs: list) -> tuple:
    """
    Run a full check-in conversation given a list of pre-scripted patient responses.
    Returns (list of assistant messages, final structured report or None).
    """
    system_prompt = build_system_prompt(patient)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"[CALL CONNECTED] Patient {patient['name']} has answered the phone. Begin the check-in call."
        }
    ]

    assistant_replies = []
    report = None

    for user_input in user_inputs:
        response = _client.chat.completions(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
        )
        assistant_msg = response.choices[0].message.content
        assistant_replies.append(assistant_msg)
        messages.append({"role": "assistant", "content": assistant_msg})

        extracted = extract_json_report(assistant_msg)
        if extracted:
            report = extracted
            break

        if user_input:
            messages.append({"role": "user", "content": user_input})

    if not report:
        messages.append({
            "role": "user",
            "content": "[CALL ENDED] The patient has hung up. Provide the structured JSON check-in report now."
        })
        response = _client.chat.completions(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.0,
        )
        final_msg = response.choices[0].message.content
        assistant_replies.append(final_msg)
        report = extract_json_report(final_msg)

    return assistant_replies, report
