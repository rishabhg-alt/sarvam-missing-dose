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
- Keep sentences SHORT, warm, and natural for a phone call (rural patients may have low health literacy)
- CRITICAL: Conduct this as an interactive turn-by-turn phone conversation. Speak only ONE or TWO sentences per turn. Ask ONE question at a time and WAIT for the patient's response. Do NOT ask all questions in one turn!
- Conversation progression:
  1. Greet the patient warmly, introduce the clinic, and ask about their first medication ({medications}).
  2. Acknowledge what they said, then check if they missed any doses or had difficulties.
  3. Ask if they have experienced any new or worsening symptoms since discharge.
  4. Remind them of their follow-up appointment ({next_appointment}) and confirm they will attend.
  5. Warmly thank them and conclude the call.

Only when the call has concluded and you have bid farewell, attach a structured JSON summary (always in English) wrapped in ```json ... ``` markers with this exact schema:
{{
  "medication_adherence": {{
    "<medication_name>": "taken" | "missed" | "partial" | "unknown"
  }},
  "symptoms_reported": ["list of symptoms mentioned, empty list if none"],
  "appointment_confirmed": true | false,
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

PATIENT_DEMO_RESPONSES = {
    "P001": [
        "नमस्ते, मैं ठीक हूँ।",
        "हाँ, मैंने सुबह मेटफ़ॉर्मिन ली थी, कोई परेशानी नहीं हुई।",
        "एमलोडिपिन भी ली, लेकिन कल रात को एक खुराक भूल गया था।",
        "मुझे थोड़ा सिरदर्द रहता है, बाकी सब ठीक है।",
        "जी हाँ, मैं 5 नवंबर को अस्पताल ज़रूर आऊंगा। बहुत-बहुत धन्यवाद।",
    ],
    "P002": [
        "வணக்கம், சொல்லுங்க.",
        "இல்லைங்க, நான் இரண்டு நாட்களாக எந்த மருந்தும் சாப்பிடவில்லை.",
        "நெஞ்சில் கொஞ்சம் பாரமாகவும், மூச்சு விட சிரமமாகவும் இருக்கு.",
        "நான் வர முயற்சிக்கிறேன், ஆனால் உடம்பு ரொம்ப பலவீனமாக இருக்கு.",
        "சரிங்க, நன்றி.",
    ],
    "P003": [
        "હા નમસ્તે, હું સારો છું.",
        "હા, એસ્પિરિન હું દરરોજ નિયમિત સવારે લઉં છું.",
        "ના, મને કોઈ નવું દર્દ કે તકલીફ નથી, ખૂબ સારું લાગે છે.",
        "હા ચોક્કસ, હું 8 નવેમ્બરે એપોઈન્ટમેન્ટ પર આવીશ.",
        "ખૂબ ખૂબ આભાર.",
    ],
}

DEFAULT_PATIENT_RESPONSES = [
    "Yes, hello. I am doing okay.",
    "Yes, I took my morning medicine.",
    "No other issues with the medicines.",
    "I have been feeling a little tired, but otherwise okay.",
    "Yes, I will definitely attend my appointment. Thank you.",
]

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

def run_turn_by_turn(patient: dict, user_inputs: list = None):
    """
    Generator that conducts an interactive turn-by-turn check-in.
    Yields:
      ("agent", spoken_text)
      ("patient", spoken_text)
      ("report", report_dict)
    """
    if user_inputs is None:
        user_inputs = PATIENT_DEMO_RESPONSES.get(patient["id"], DEFAULT_PATIENT_RESPONSES)

    system_prompt = build_system_prompt(patient)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"[CALL CONNECTED] Patient {patient['name']} answered the call. Please introduce yourself and start the check-in."
        }
    ]

    report = None

    for i, user_reply in enumerate(user_inputs):
        response = _client.chat.completions(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        assistant_full = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_full})

        extracted = extract_json_report(assistant_full)
        spoken_agent = assistant_full.split("```json")[0].strip()

        if spoken_agent:
            yield ("agent", spoken_agent)

        if extracted:
            report = extracted
            break

        # Patient speaks next
        if user_reply:
            yield ("patient", user_reply)
            messages.append({"role": "user", "content": user_reply})

    if not report:
        messages.append({
            "role": "user",
            "content": "[CALL CONCLUDED] The conversation has ended. Provide the structured JSON check-in report now."
        })
        response = _client.chat.completions(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.0,
        )
        final_msg = response.choices[0].message.content
        report = extract_json_report(final_msg)

    if report:
        yield ("report", report)

def run_text_conversation(patient: dict, user_inputs: list = None) -> tuple:
    assistant_replies = []
    report = None
    for item_type, content in run_turn_by_turn(patient, user_inputs):
        if item_type == "agent":
            assistant_replies.append(content)
        elif item_type == "report":
            report = content
    return assistant_replies, report
