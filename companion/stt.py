"""Speech-to-Text wrapper using Sarvam Saaras. Accepts audio bytes, returns transcript."""
import io
import os
from dotenv import load_dotenv
from sarvamai import SarvamAI
from companion.config import STT_MODEL

load_dotenv()
_client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

def transcribe(audio_bytes: bytes, language_code: str) -> str:
    """Transcribe audio bytes to text."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    response = _client.speech_to_text.transcribe(
        file=audio_file,
        model="saaras:v2",
        language_code=language_code,
        mode="transcribe",
    )
    return response.transcript
