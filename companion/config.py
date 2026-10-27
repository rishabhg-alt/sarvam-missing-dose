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

# Language -> Bulbul speaker (medical/warm tone, confirmed to exist in SDK)
VOICE_MAP = {
    "hi-IN": "ritu_hi_medical",
    "ta-IN": "kavitha_ta_conversation",
    "te-IN": "kavitha_te_conversation",
    "bn-IN": "roopa_bn_conversational",
    "mr-IN": "pooja_mr_conversational",
    "gu-IN": "pooja_gu_conversational",
    "kn-IN": "kavitha_kn_conversation",
    "ml-IN": "kavitha_ml_conversation",
    "pa-IN": "simran",
    "en-IN": "ishita_en_medical",
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
