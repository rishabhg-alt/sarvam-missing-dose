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

PATIENT_CLINICAL_PROFILES = {
    "P001": {
        "persona": "Ramesh Kumar, a polite rural Indian gentleman speaking in Hindi.",
        "facts": (
            "- General state: Feeling fine / okay.\n"
            "- Metformin: Took morning Metformin 500mg without any trouble.\n"
            "- Amlodipine: Took it, but forgot last night's dose.\n"
            "- Symptoms: Has a mild headache, but no other symptoms.\n"
            "- Follow-up appointment (5 Nov): Confirms he will definitely attend.\n"
            "- Additional help / questions: Does not need any transport help, has no questions.\n"
            "- Farewell: If caller concludes or says goodbye, politely say thank you and goodbye ('जी बहुत धन्यवाद, नमस्ते!')."
        ),
        "fallbacks": {
            "greeting": "नमस्ते, मैं ठीक हूँ।",
            "medication": "हाँ, मैंने सुबह मेटफ़ॉर्मिन ली थी, लेकिन कल रात एमलोडिपिन भूल गया था।",
            "symptoms": "मुझे थोड़ा सिरदर्द रहता है, बाकी सब ठीक है।",
            "appointment": "जी हाँ, मैं 5 नवंबर को अस्पताल ज़रूर आऊंगा। बहुत-बहुत धन्यवाद।",
            "questions": "नहीं, मुझे किसी मदद या सवाल की ज़रूरत नहीं है, धन्यवाद।",
            "farewell": "जी बहुत धन्यवाद, नमस्ते!",
        }
    },
    "P002": {
        "persona": "Lakshmi Devi, an elderly rural patient speaking in Tamil.",
        "facts": (
            "- General state: Feeling weak and unwell.\n"
            "- Medications (Atorvastatin, Losartan): Has NOT taken any medicines for the last two days.\n"
            "- Symptoms: Chest feels heavy, having difficulty breathing; it is new and getting slightly worse.\n"
            "- Follow-up appointment (10 Nov): Wants to come, but body is very weak.\n"
            "- Additional help / questions: Needs help / transport from hospital to travel.\n"
            "- Farewell: If caller concludes or says goodbye, say thank you ('சரிங்க, ரொம்ப நன்றிங்க!')."
        ),
        "fallbacks": {
            "greeting": "வணக்கம், சொல்லுங்க.",
            "medication": "இல்லைங்க, நான் இரண்டு நாட்களாக எந்த மருந்தும் சாப்பிடவில்லை.",
            "symptoms": "நெஞ்சில் கொஞ்சம் பாரமாகவும், மூச்சு விட சிரமமாகவும் இருக்கு.",
            "appointment": "நான் வர முயற்சிக்கிறேன், ஆனால் உடம்பு ரொம்ப பலவீனமாக இருக்கு.",
            "questions": "ஆமாங்க, எனக்கு ஆஸ்பத்திரிக்கு போக உதவி தேவைப்படும்.",
            "farewell": "சரிங்க, ரொம்ப நன்றிங்க!",
        }
    },
    "P003": {
        "persona": "Suresh Patel, a cheerful rural patient speaking in Gujarati.",
        "facts": (
            "- General state: Doing very well.\n"
            "- Aspirin: Takes Aspirin 75mg regularly every morning without fail.\n"
            "- Symptoms: No pain, no new symptoms, feeling completely healthy.\n"
            "- Follow-up appointment (8 Nov): Confirms he will definitely come.\n"
            "- Additional help / questions: Has no other questions and no difficulties.\n"
            "- Farewell: If caller concludes or says goodbye, say thank you ('ખૂબ ખૂબ આભાર, આવજો!')."
        ),
        "fallbacks": {
            "greeting": "હા નમસ્તે, હું સારો છું.",
            "medication": "હા, એસ્પિરિન હું દરરોજ નિયમિત સવારે લઉં છું, કોઈ ડોઝ ચૂક્યો નથી.",
            "symptoms": "ના, મને કોઈ નવું દર્દ કે તકલીફ નથી, ખૂબ સારું લાગે છે.",
            "appointment": "હા ચોક્કસ, હું 8 નવેમ્બરે એપોઈન્ટમેન્ટ પર આવીશ.",
            "questions": "ના, મને કોઈ પ્રશ્ન કે મુશ્કેલી નથી, આભાર.",
            "farewell": "ખૂબ ખૂબ આભાર, આવજો!",
        }
    }
}

DEFAULT_PATIENT_RESPONSES = [
    "Yes, hello. I am doing okay.",
    "Yes, I took my morning medicine.",
    "No other issues with the medicines.",
    "I have been feeling a little tired, but otherwise okay.",
    "Yes, I will definitely attend my appointment. Thank you.",
]

def is_call_conclusion(text: str) -> bool:
    """Detect if the agent is bidding farewell or concluding the conversation."""
    t = (text or "").lower()
    closing_indicators = [
        "ख्याल रखिएगा", "ध्यान रखिएगा", "ख्याल रखिए", "ध्यान रखिए", "अलविदा", "स्वस्थ रहें", "प्रणाम",
        "நல்லா பார்த்துக்கோங்க", "கவனமா இருங்க", "போயிட்டு வரேன்", "உடம்பை பார்த்துக்கோங்க",
        "સાચવજો", "ધ્યાન રાખજો", "તમારું ધ્યાન",
        "take care", "goodbye", "have a good day", "have a great day"
    ]
    return any(ind in t for ind in closing_indicators)

def get_or_create_patient_profile(patient: dict) -> dict:
    pid = patient.get("id", "")
    if pid in PATIENT_CLINICAL_PROFILES:
        return PATIENT_CLINICAL_PROFILES[pid]

    name = patient.get("name", "Patient")
    gender = patient.get("gender", "male")
    lang = patient.get("language", "hi-IN")
    lang_name = LANGUAGE_NAMES.get(lang, "Hindi")
    meds = ", ".join(patient.get("medications", []))
    appt = patient.get("next_appointment", "")

    return {
        "persona": f"{name}, a {gender} rural patient speaking in {lang_name}.",
        "facts": (
            f"- General state: Doing well / okay.\n"
            f"- Medications ({meds}): Has been taking prescribed medicines regularly.\n"
            f"- Symptoms: Feeling okay, no severe symptoms.\n"
            f"- Follow-up appointment ({appt}): Confirms attendance.\n"
            f"- Farewell: Say thank you and goodbye politely."
        ),
        "fallbacks": {
            "greeting": "Hello, I am doing fine.",
            "medication": "Yes, I am taking my medicines regularly.",
            "symptoms": "No new symptoms, feeling fine.",
            "appointment": f"Yes, I will attend my appointment on {appt}.",
            "questions": "No questions, thank you.",
            "farewell": "Thank you very much, goodbye!",
        }
    }

def get_fallback_patient_reply(patient: dict, agent_message: str, turn_index: int) -> str:
    profile = get_or_create_patient_profile(patient)
    fallbacks = profile.get("fallbacks", {})
    t = agent_message.lower()

    if is_call_conclusion(t) or "अलविदा" in t:
        return fallbacks.get("farewell", "जी बहुत धन्यवाद, नमस्ते!")
    if any(k in t for k in ["सवाल", "प्रश्न", "கேள்வி", "question", "गाड़ी", "वाहन", "मदद", "உதவி"]):
        return fallbacks.get("questions", "नहीं, मुझे कोई और मदद या सवाल नहीं है, धन्यवाद।")
    if any(k in t for k in ["अपॉइंटमेंट", "appointment", "तारीख", "तारीખ", "தேதி", "5 नवंबर", "10 nov", "8 nov", "8 નવેમ્બર", "நவம்பர்"]):
        return fallbacks.get("appointment", fallbacks.get("farewell"))
    if any(k in t for k in ["लक्षण", "दर्द", "सिरदर्द", "सांस", "நெஞ்சு", "வலி", "તકલીફ", "દર્દ", "symptom"]):
        return fallbacks.get("symptoms", fallbacks.get("farewell"))
    if any(k in t for k in ["दवा", "गोली", "खुराक", "मेटफॉर्मिन", "एमलोडिपिन", "மருந்து", "மாத்திரை", "દવા", "ગોળી", "medicine"]):
        return fallbacks.get("medication", fallbacks.get("greeting"))

    return fallbacks.get("farewell", "जी बहुत धन्यवाद, नमस्ते!")

def generate_patient_reply(patient: dict, agent_message: str, turn_index: int) -> str:
    """Dynamically generate context-relevant patient reply directly answering agent's query."""
    profile = get_or_create_patient_profile(patient)
    lang = patient.get("language", "hi-IN")
    lang_name = LANGUAGE_NAMES.get(lang, "Hindi")

    # If caller is concluding call or saying goodbye, return quick polite farewell
    if is_call_conclusion(agent_message):
        if "hi" in lang:
            return "जी बहुत धन्यवाद, नमस्ते!"
        elif "ta" in lang:
            return "சரிங்க, ரொம்ப நன்றிங்க!"
        elif "gu" in lang:
            return "ખૂબ ખૂબ આભાર, આવજો!"
        return "Thank you so much, goodbye!"

    # Call Sarvam AI for dynamic, highly relevant contextual response
    try:
        persona = profile["persona"]
        facts = profile["facts"]
        sys_msg = (
            f"You are {persona}\n"
            f"Your medical reality:\n{facts}\n\n"
            f"STRICT RULES:\n"
            f"- Speak in {lang_name} ONLY.\n"
            f"- Reply in 1 short, natural spoken sentence directly answering what the clinic caller just asked or said.\n"
            f"- Answer ONLY the specific question asked. Do NOT volunteer facts about other unasked topics.\n"
            f"- Never output JSON, markdown, or greetings if already greeted."
        )
        resp = _client.chat.completions(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"[Clinic caller asks]: {agent_message}"}
            ],
            temperature=0.1,
            max_tokens=60,
        )
        reply = (resp.choices[0].message.content or "").strip()
        reply = reply.split("\n")[0].strip()
        reply = re.sub(r'^["\']|["\']$', '', reply)
        if reply and len(reply) > 2 and not reply.startswith("{"):
            return reply
    except Exception as e:
        print(f"Dynamic patient generation exception (using fallback): {e}")

    return get_fallback_patient_reply(patient, agent_message, turn_index)

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
    max_turns = len(user_inputs) if user_inputs is not None else 6

    for turn_idx in range(max_turns):
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
            # If agent bid farewell and report was attached, patient acknowledges goodbye
            if spoken_agent and is_call_conclusion(spoken_agent):
                farewell_ack = generate_patient_reply(patient, spoken_agent, turn_idx)
                yield ("patient", farewell_ack)
            break

        # Check if agent bid farewell without attaching report yet
        agent_closed = spoken_agent and is_call_conclusion(spoken_agent)

        # Patient speaks next, directly answering what the agent asked
        if user_inputs is not None and turn_idx < len(user_inputs):
            patient_reply = user_inputs[turn_idx]
        else:
            patient_reply = generate_patient_reply(patient, spoken_agent, turn_idx)

        if patient_reply:
            yield ("patient", patient_reply)
            messages.append({"role": "user", "content": patient_reply})

        # If the agent concluded the call, break after patient acknowledged goodbye
        if agent_closed:
            break

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
