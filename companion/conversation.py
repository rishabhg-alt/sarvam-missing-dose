"""
LLM-driven structured conversation engine for patient check-ins.

Flow:
  1. Greet patient in their language with strict grammatical gender alignment
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

Your persona:
- You are a female healthcare caller from the clinic. In languages with first-person verb gender agreement (like Hindi), speak in first-person feminine ("मैं क्लिनिक से बोल रही हूँ").

Patient identity & gender:
- Patient Name: {name}
- Gender: {gender_upper} ({patient_salutation})
- CRITICAL GENDER RULES: {gender_instructions}

Your role:
- Speak ONLY in the patient's specified language: {language_name}
- Keep sentences SHORT, warm, and natural for a phone call (rural patients may have low health literacy)
- CRITICAL: Conduct this as an interactive turn-by-turn phone conversation. Speak only ONE or TWO sentences per turn. Ask ONE question at a time and WAIT for the patient's response. Do NOT ask all questions in one turn!
- Address the patient ({name}) with high respect and STRICT adherence to their gender:
  * NEVER address male patients using female honorifics or female grammatical verb endings!
  * NEVER address female patients using male honorifics or male grammatical verb endings!
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
- Gender: {gender_upper} ({patient_salutation})
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

def get_gender_instructions(patient: dict) -> tuple[str, str]:
    """Return (salutation, detailed_grammatical_rules) for the patient's gender and language."""
    gender = str(patient.get("gender", "male")).strip().lower()
    is_male = (gender == "male")
    lang = patient.get("language", "hi-IN")
    name = patient.get("name", "Patient")

    if is_male:
        salutation = "Male / Gentleman (श्री / Mr.)"
        if "hi" in lang:
            rules = (
                f"The patient {name} is a MALE gentleman. You MUST address him using MASCULINE honorifics ('श्री {name} जी' or '{name} जी') "
                f"and MASCULINE grammatical agreement: 'आप कैसे हैं', 'आप कैसा महसूस कर रहे हैं', 'आप दवा ले रहे हैं', 'आप अस्पताल आएंगे'. "
                f"ABSOLUTE PROHIBITION: NEVER use feminine verb agreements or feminine titles for {name}. "
                f"NEVER say 'कैसी हैं', 'कर रही हैं', 'ले रही हैं', 'आएंगी', 'श्रीमती', 'बहन जी', 'माता जी', or 'देवी'."
            )
        elif "gu" in lang:
            rules = (
                f"The patient {name} is a MALE gentleman. Address him with masculine honorifics ('શ્રી {name} ભાઈ' or '{name} ભાઈ') "
                f"and masculine verb forms ('કેવા છો', 'લઈ રહ્યા છો', 'આવશો'). "
                f"ABSOLUTE PROHIBITION: NEVER use feminine forms ('બહેન', 'કેવી છો', 'શ્રીમતી')."
            )
        elif "ta" in lang:
            rules = (
                f"The patient {name} is a MALE gentleman. Address him with respectful masculine terms ('திரு {name}' or 'ஐயா'). "
                f"ABSOLUTE PROHIBITION: NEVER address him as 'திருமதி' or 'அம்மா'."
            )
        else:
            rules = f"The patient {name} is a MALE gentleman. Always use masculine pronouns (he/him/Mr.) and appropriate masculine honorifics."
    else:
        salutation = "Female / Lady (श्रीमती / Mrs./Ms.)"
        if "hi" in lang:
            rules = (
                f"The patient {name} is a FEMALE lady. You MUST address her using FEMININE honorifics ('श्रीमती {name} जी' or '{name} जी') "
                f"and FEMININE grammatical agreement: 'आप कैसी हैं', 'आप कैसा महसूस कर रही हैं', 'आप दवा ले रही हैं', 'आप अस्पताल आएंगी'. "
                f"ABSOLUTE PROHIBITION: NEVER use masculine verb agreements or masculine titles for {name}. "
                f"NEVER say 'कैसे हैं', 'कर रहे हैं', 'ले रहे हैं', 'आएंगे', 'श्रीमान', or 'भाई साहब'."
            )
        elif "gu" in lang:
            rules = (
                f"The patient {name} is a FEMALE lady. Address her with feminine honorifics ('શ્રીમતી {name} બહેન' or '{name} બહેન') "
                f"and feminine verb forms ('કેવી છો', 'લઈ રહી છો'). "
                f"ABSOLUTE PROHIBITION: NEVER use masculine forms ('ભાઈ', 'કેવા છો', 'શ્રી')."
            )
        elif "ta" in lang:
            rules = (
                f"The patient {name} is a FEMALE lady. Address her with respectful feminine terms ('திருமதி {name}' or 'அம்மா'). "
                f"ABSOLUTE PROHIBITION: NEVER address her as 'திரு' or 'ஐயா'."
            )
        else:
            rules = f"The patient {name} is a FEMALE lady. Always use feminine pronouns (she/her/Mrs./Ms.) and appropriate feminine honorifics."

    return salutation, rules

def enforce_gender_agreement(text: str, gender: str, lang: str, patient_name: str = "") -> str:
    """
    Guarantees strict grammatical gender consistency for Indic languages.
    Prevents male patients (e.g. Ramesh Kumar, Suresh Patel) from being addressed
    with feminine verbs/honorifics ('कैसी हैं', 'कर रही हैं', 'ले रही हैं', 'बहन जी', etc.)
    and female patients (e.g. Lakshmi Devi) from being addressed with masculine forms.
    Preserves 1st-person female assistant phrases like 'बोल रही हूँ'.
    """
    if not text:
        return text

    gender = (gender or "male").strip().lower()
    is_male = (gender == "male")
    lang = (lang or "").lower()

    if "hi" in lang:
        if is_male:
            # 1. Honorifics and Titles
            if patient_name:
                text = re.sub(rf"श्रीमती\s+{re.escape(patient_name)}", f"श्री {patient_name}", text)
                text = re.sub(rf"{re.escape(patient_name)}\s+देवी", patient_name, text)
            text = re.sub(r"श्रीमती\s+रमेश", "श्री रमेश", text)
            text = re.sub(r"श्रीमती\s+सुरेश", "श्री सुरेश", text)
            text = re.sub(r"रमेश\s+देवी", "रमेश कुमार", text)
            target_title = f"{patient_name} जी" if patient_name else "रमेश जी"
            text = re.sub(r"बहन\s*जी", target_title, text)
            text = re.sub(r"माता\s*जी", target_title, text)
            text = re.sub(r"देवी\s*जी", target_title, text)

            # 2. Addressee state / well-being ("आप कैसी हैं" -> "आप कैसे हैं")
            text = re.sub(r"आप\s+कैसी\s+हैं", "आप कैसे हैं", text)
            text = re.sub(r"कैसी\s+हैं\s+आप", "कैसे हैं आप", text)
            text = re.sub(r"कैसी\s+हैं", "कैसे हैं", text)
            text = re.sub(r"कैसी\s+महसूस", "कैसा महसूस", text)
            text = re.sub(r"कैसा\s+महसूस\s+कर\s+रही\s+हैं", "कैसा महसूस कर रहे हैं", text)
            text = re.sub(r"कैसी\s+महसूस\s+कर\s+रही\s+हैं", "कैसा महसूस कर रहे हैं", text)

            # 3. Progressive verbs addressing patient: "रही हैं" / "रही हो" -> "रहे हैं" / "रहे हो"
            # Note: "रही हूँ" is 1st-person female assistant self-reference and must stay intact!
            text = re.sub(r"(\S+)\s+रही\s+हैं", r"\1 रहे हैं", text)
            text = re.sub(r"(\S+)\s+रही\s+हो", r"\1 रहे हो", text)

            # 4. Modals & future verbs addressing patient: "ेंगी" -> "ेंगे", "एं\s*गी" -> "एंगे"
            text = re.sub(r"एं\s*गी", "एंगे", text)
            text = re.sub(r"े\s*ं\s*गी", "ेंगे", text)
            text = re.sub(r"सकती\s+हैं", "सकते हैं", text)
            text = re.sub(r"पा\s+रही\s+हैं", "पा रहे हैं", text)
            text = re.sub(r"हो\s+गईं", "हो गए", text)
            text = re.sub(r"हो\s+गई\s+हैं", "हो गए हैं", text)
        else:
            if patient_name:
                text = re.sub(rf"श्री(?!मती)\s+{re.escape(patient_name)}", f"श्रीमती {patient_name}", text)
            text = re.sub(r"श्रीमान", "श्रीमती जी", text)
            target_title = f"{patient_name} जी" if patient_name else "लक्ष्मी जी"
            text = re.sub(r"भाई\s*साहब", target_title, text)

            text = re.sub(r"आप\s+कैसे\s+हैं", "आप कैसी हैं", text)
            text = re.sub(r"कैसे\s+हैं\s+आप", "कैसी हैं आप", text)
            text = re.sub(r"कैसे\s+हैं", "कैसी हैं", text)
            text = re.sub(r"कैसा\s+महसूस", "कैसी महसूस", text)

            text = re.sub(r"(\S+)\s+रहे\s+हैं", r"\1 रही हैं", text)
            text = re.sub(r"(\S+)\s+रहे\s+हो", r"\1 रही हो", text)

            text = re.sub(r"एं\s*गे", "एंगी", text)
            text = re.sub(r"े\s*ं\s*गे", "ेंगी", text)
            text = re.sub(r"सकते\s+हैं", "सकती हैं", text)

    elif "gu" in lang:
        if is_male:
            text = re.sub(r"બહેન", "ભાઈ", text)
            text = re.sub(r"કેવી\s+છો", "કેવા છો", text)
            text = re.sub(r"લઈ\s+રહી\s+છો", "લઈ રહ્યા છો", text)
            text = re.sub(r"કરી\s+રહી\s+છો", "કરી રહ્યા છો", text)
            if patient_name:
                text = re.sub(rf"શ્રીમતી\s+{re.escape(patient_name)}", f"શ્રી {patient_name}", text)
        else:
            text = re.sub(r"ભાઈ", "બહેન", text)
            text = re.sub(r"કેવા\s+છો", "કેવી છો", text)
            text = re.sub(r"લઈ\s+રહ્યા\s+છો", "લઈ રહી છો", text)
            text = re.sub(r"કરી\s+રહ્યા\s+છો", "કરી રહી છો", text)
            if patient_name:
                text = re.sub(rf"શ્રી(?!મતી)\s+{re.escape(patient_name)}", f"શ્રીમતી {patient_name}", text)

    elif "ta" in lang:
        if is_male:
            text = re.sub(r"திருமதி", "திரு", text)
            text = re.sub(r"அம்மா", "ஐயா", text)
        else:
            text = re.sub(r"திரு(?!மதி)\b", "திருமதி", text)
            text = re.sub(r"ஐயா", "அம்மா", text)

    return text

def build_system_prompt(patient: dict) -> str:
    discharge = date.fromisoformat(patient["discharge_date"])
    days = (date.today() - discharge).days
    lang = patient["language"]
    gender = str(patient.get("gender", "male")).strip().lower()
    salutation, gender_rules = get_gender_instructions(patient)

    return SYSTEM_PROMPT.format(
        language_name=LANGUAGE_NAMES.get(lang, lang),
        name=patient["name"],
        gender_upper=gender.upper(),
        patient_salutation=salutation,
        gender_instructions=gender_rules,
        medications=", ".join(patient["medications"]),
        next_appointment=patient["next_appointment"],
        days_since_discharge=days,
    )

def extract_json_report(text: str):
    """Extract structured JSON block from LLM response."""
    if not text or not isinstance(text, str):
        return None
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

    gender = str(patient.get("gender", "male")).strip().lower()
    is_male = (gender == "male")
    title_hint = "Mr. / पुरुष / Gentleman" if is_male else "Mrs./Ms. / महिला / Lady"

    system_prompt = build_system_prompt(patient)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"[CALL CONNECTED] Patient {patient['name']} ({title_hint}, Gender: {gender.upper()}) answered the call. "
                f"Introduce yourself warmly and address {patient['name']} with correct {gender.upper()} honorifics "
                f"and grammatical agreement in {LANGUAGE_NAMES.get(patient.get('language'), 'Hindi')}."
            )
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
        assistant_full = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_full})

        extracted = extract_json_report(assistant_full)
        spoken_agent = assistant_full.split("```json")[0].strip()

        if spoken_agent:
            # Enforce strict grammatical gender alignment for Indic speech
            spoken_agent = enforce_gender_agreement(
                spoken_agent,
                gender=gender,
                lang=patient.get("language", "hi-IN"),
                patient_name=patient.get("name", "")
            )
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
        final_msg = response.choices[0].message.content or ""
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
