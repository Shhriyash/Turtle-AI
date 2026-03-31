from __future__ import annotations

import os

from groq import Groq


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY or GROQ_API_KEY2 for Groq TTS fallback.")
    return Groq(api_key=api_key)


def get_deepgram_client():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPGRAM_API_KEY for Deepgram TTS.")

    try:
        from deepgram import DeepgramClient
    except Exception as exc:
        raise RuntimeError("deepgram-sdk is not available in this environment.") from exc

    return DeepgramClient(api_key=api_key)
