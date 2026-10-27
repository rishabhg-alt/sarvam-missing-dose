"""
Entry point: run a structured check-in for one or all patients.

Usage:
    python run_checkup.py              # Check in all patients (demo mode)
    python run_checkup.py P001         # Check in patient P001 (demo mode)
    python run_checkup.py P001 --audio # Check in P001 and play TTS audio (macOS)
"""
import sys
from companion.call_simulator import simulate_call
from companion import patient_db

# Pre-scripted demo responses per patient
DEMO_RESPONSES = {
    "P001": [
        "हाँ, मैंने आज सुबह Metformin ली थी।",
        "Amlodipine भी ली, लेकिन कल रात भूल गया था।",
        "थोड़ा सिरदर्द है, बाकी ठीक हूँ।",
        "हाँ, मैं अपॉइंटमेंट पर जरूर आऊंगा।",
    ],
    "P002": [
        "வணக்கம். நான் நேற்று மருந்து சாப்பிடவில்லை.",
        "இரண்டு நாட்களாக மருந்து எடுக்கவில்லை.",
        "நெஞ்சில் கொஞ்சம் அசௌகரியமாக உள்ளது.",
        "Appointment-க்கு வர முயற்சிக்கிறேன்.",
    ],
    "P003": [
        "હા, હું Aspirin દરરોજ સવારે લઉં છું.",
        "કોઈ નવા લક્ષણ નથી, સારું લાગી રહ્યું છે.",
        "હા, ચોક્કસ appointment પર આવીશ.",
    ],
}

DEFAULT_RESPONSES = [
    "Yes, I am taking my medicines regularly.",
    "No new symptoms, I feel okay.",
    "Yes, I will attend the appointment.",
]

def main():
    args = sys.argv[1:]
    play_audio = "--audio" in args
    patient_id = next((a for a in args if not a.startswith("--")), None)

    patient_db.load_patients()

    if patient_id:
        patients = [patient_db.get_patient(patient_id)]
        if not patients[0]:
            print(f"❌ Patient {patient_id} not found.")
            sys.exit(1)
    else:
        patients = patient_db.get_all_patients()

    for patient in patients:
        pid = patient["id"]
        responses = DEMO_RESPONSES.get(pid, DEFAULT_RESPONSES)
        simulate_call(pid, patient_responses=responses, play_audio=play_audio)
        print("\n" + "─"*60 + "\n")

if __name__ == "__main__":
    main()
