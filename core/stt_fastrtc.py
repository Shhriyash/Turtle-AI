from __future__ import annotations

import io
import os
from typing import Tuple
import wave

import numpy as np
from groq import AsyncGroq

from tools.tts.client import get_async_groq_client


class FastRTCSTT:
    """Simple STT adapter used by the voice app.

    Accepts `(sample_rate, np.ndarray)` audio tuples and returns Whisper text.

    Transcription is async (``AsyncGroq``) so cancelling the awaiting task closes
    the HTTP request. It used to be a sync call parked in an executor thread,
    which cannot be cancelled. With no injected client the process-wide
    per-loop ``AsyncGroq`` from ``tools.tts.client`` is used (built once, with
    explicit timeouts) instead of constructing a client per instance.
    """

    def __init__(
        self,
        groq_client: AsyncGroq | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._client = groq_client
        self._api_key = api_key
        self.model = model or os.getenv("STT_MODEL", "whisper-large-v3-turbo")

    @property
    def client(self) -> AsyncGroq:
        return self._client or get_async_groq_client(self._api_key)

    async def transcribe_from_audio(self, audio: Tuple[int, np.ndarray]) -> str:
        sample_rate, audio_array = audio
        if audio_array is None or len(audio_array) == 0:
            return ""

        valid_rate = max(8000, min(int(sample_rate), 48000))
        audio_data = np.array(audio_array, dtype=np.int16)

        # Keep everything in-memory to avoid Windows temp-file lock races.
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # int16 PCM
            wav_file.setframerate(valid_rate)
            wav_file.writeframes(audio_data.tobytes())

        transcription = await self.client.audio.transcriptions.create(
            file=("input.wav", wav_buffer.getvalue()),
            model=self.model,
            response_format="verbose_json",
        )
        return getattr(transcription, "text", "") or ""
