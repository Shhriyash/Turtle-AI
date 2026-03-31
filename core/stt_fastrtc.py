from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from groq import Groq


class FastRTCSTT:
    """Simple STT adapter used by the voice app.

    Accepts `(sample_rate, np.ndarray)` audio tuples and returns Whisper text.
    """

    def __init__(self, groq_client: Groq | None = None) -> None:
        self.client = groq_client or Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2"))

    def transcribe_from_audio(self, audio: Tuple[int, np.ndarray]) -> str:
        sample_rate, audio_array = audio
        if audio_array is None or len(audio_array) == 0:
            return ""

        valid_rate = max(8000, min(int(sample_rate), 48000))
        audio_data = np.array(audio_array, dtype=np.int16)

        temp_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            import scipy.io.wavfile as wavfile

            wavfile.write(str(temp_path), valid_rate, audio_data)
            with open(temp_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(temp_path.name, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
            return getattr(transcription, "text", "") or ""
        finally:
            temp_path.unlink(missing_ok=True)
