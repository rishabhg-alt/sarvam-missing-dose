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

# Language -> Bulbul speaker for Agent (calm, warm tone compatible with bulbul:v3)
VOICE_MAP = {
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

# Language -> Bulbul speaker for Patient (distinct tone compatible with bulbul:v3)
PATIENT_VOICE_MAP = {
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
