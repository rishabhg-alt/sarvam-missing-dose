"""Text-to-Speech wrapper using Sarvam Bulbul. Returns base64-decoded audio bytes."""
import base64
import os
from dotenv import load_dotenv
from sarvamai import SarvamAI
from companion.config import VOICE_MAP, TTS_MODEL

load_dotenv()
_client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

def speak(text: str, language_code: str, speaker: str = None) -> bytes:
    """Convert text to audio bytes using Bulbul TTS."""
    if not speaker:
        speaker = VOICE_MAP.get(language_code, "ritu")
    response = _client.text_to_speech.convert(
        text=text,
        language_code=language_code,
        speaker=speaker,
        model=TTS_MODEL,
        speech_sample_rate=8000,
        output_audio_codec="wav",
    )
    audio_b64 = response.audios[0]
    return base64.b64decode(audio_b64)
