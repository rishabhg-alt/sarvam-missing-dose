"""Central configuration: language names, voice mappings, escalation thresholds."""

LANGUAGE_NAMES = {
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "pa-IN": "Punjabi",
    "en-IN": "English",
}

# Language -> Female Bulbul speakers for Agent & Female Patients
VOICE_MAP_FEMALE = {
    "hi-IN": "ritu",
    "ta-IN": "kavitha",
    "te-IN": "kavitha",
    "bn-IN": "roopa",
    "mr-IN": "pooja",
    "gu-IN": "pooja",
    "kn-IN": "kavitha",
    "ml-IN": "kavitha",
    "pa-IN": "simran",
    "en-IN": "ishita",
}

# Language -> Male Bulbul speakers for Male Patients & Male Agents
VOICE_MAP_MALE = {
    "hi-IN": "shubh",
    "ta-IN": "vijay",
    "te-IN": "vijay",
    "bn-IN": "amit",
    "mr-IN": "rahul",
    "gu-IN": "aditya",
    "kn-IN": "mani",
    "ml-IN": "mani",
    "pa-IN": "kabir",
    "en-IN": "aditya",
}

# Backward-compatibility aliases
VOICE_MAP = VOICE_MAP_FEMALE
PATIENT_VOICE_MAP = VOICE_MAP_MALE

def get_speaker_for_patient(patient: dict) -> str:
    """Return gender-matching voice for patient."""
    lang = patient.get("language", "hi-IN")
    gender = str(patient.get("gender", "male")).lower()
    if gender == "female":
        return VOICE_MAP_FEMALE.get(lang, "kavitha")
    return VOICE_MAP_MALE.get(lang, "shubh")

def get_speaker_for_agent(lang: str) -> str:
    """Clinic assistant uses a warm, empathetic female voice."""
    return VOICE_MAP_FEMALE.get(lang, "ritu")

TTS_MODEL = "bulbul:v3"
STT_MODEL = "saaras:v2"
CHAT_MODEL = "sarvam-105b-conversations"

ESCALATION_TRIGGERS = [
    "missed_critical_dose",
    "severe_symptom_reported",
    "missed_appointment",
    "patient_confused",
    "emergency_symptom",
]
