"""
FastRTC-style STT helper shared by console assistants.

Uses the same audio preparation flow as fastrtc_real to normalize audio
before sending it to Groq Whisper.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
from groq import Groq


class FastRTCSTT:
    """STT helper that normalizes audio and transcribes via Groq Whisper."""

    def __init__(self, groq_client: Groq | None = None, model: str = "whisper-large-v3-turbo") -> None:
        self.groq_client = groq_client or Groq(
            api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
        )
        self.model = model

    def transcribe_audio_file(self, audio_path: Path) -> Optional[str]:
        """Transcribe an existing audio file."""
        try:
            with open(audio_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(audio_path.name, file.read()),
                    model=self.model,
                    response_format="verbose_json",
                )
            return transcription.text
        except Exception:
            return None

    def transcribe_from_audio(self, audio: Tuple[int, np.ndarray]) -> Optional[str]:
        """Normalize a (sample_rate, audio_array) tuple and transcribe it."""
        sample_rate, audio_array = audio
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            valid_sample_rate = max(8000, min(sample_rate, 48000))
            audio_data = np.array(audio_array, dtype=np.int16)
            wavfile.write(temp_path, valid_sample_rate, audio_data)

            transcription = self.transcribe_audio_file(temp_path)
            temp_path.unlink(missing_ok=True)
            return transcription
        except Exception:
            return None
