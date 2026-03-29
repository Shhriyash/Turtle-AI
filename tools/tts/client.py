from __future__ import annotations

import os
from typing import Optional

from deepgram import DeepgramClient
from groq import Groq
from core.env import load_env

_deepgram_client: Optional[DeepgramClient] = None
_groq_client: Optional[Groq] = None


def get_deepgram_client() -> DeepgramClient:
    global _deepgram_client
    if _deepgram_client is not None:
        return _deepgram_client
    load_env()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("No Deepgram API key found. Set DEEPGRAM_API_KEY in .env.")
    _deepgram_client = DeepgramClient(api_key=api_key)
    return _deepgram_client


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    load_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("No Groq API key found. Set GROQ_API_KEY in .env.")
    _groq_client = Groq(api_key=api_key)
    return _groq_client
