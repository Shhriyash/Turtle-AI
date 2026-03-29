from __future__ import annotations

import os
from pathlib import Path
from tools.tts.client import get_deepgram_client, get_groq_client

DEEPGRAM_TTS_DEFAULT_MODEL = "aura-2-orion-en"
DEEPGRAM_TTS_DEFAULT_ENCODING = "linear16"
DEEPGRAM_TTS_DEFAULT_CONTAINER = "wav"
DEEPGRAM_TTS_DEFAULT_SAMPLE_RATE = "24000"
GROQ_TTS_DEFAULT_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_DEFAULT_VOICE = "orion"
GROQ_TTS_DEFAULT_FORMAT = "wav"

def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return any(token in message for token in ["rate limit", "rate_limit", "quota", "insufficient"])


def _synthesize_deepgram(text: str, output_path: Path, *, model: str | None = None) -> Path:
    client = get_deepgram_client()
    model_name = model or os.getenv("DEEPGRAM_TTS_MODEL", DEEPGRAM_TTS_DEFAULT_MODEL)
    encoding = os.getenv("DEEPGRAM_TTS_ENCODING", DEEPGRAM_TTS_DEFAULT_ENCODING)
    container = os.getenv("DEEPGRAM_TTS_CONTAINER", DEEPGRAM_TTS_DEFAULT_CONTAINER)
    sample_rate = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", DEEPGRAM_TTS_DEFAULT_SAMPLE_RATE))

    if output_path.suffix.lower() != f".{container}":
        output_path = output_path.with_suffix(f".{container}")

    response = client.speak.v1.audio.generate(
        text=text,
        model=model_name,
        encoding=encoding,
        container=container,
        sample_rate=sample_rate,
    )
    if hasattr(response, "stream"):
        output_path.write_bytes(response.stream.getvalue())
        return output_path

    if hasattr(response, "__iter__"):
        with open(output_path, "wb") as audio_file:
            for chunk in response:
                if chunk:
                    audio_file.write(chunk)
        return output_path

    raise RuntimeError("Deepgram TTS returned unsupported response type.")
    return output_path


def _synthesize_groq(
    text: str,
    output_path: Path,
    *,
    model: str | None = None,
    voice: str | None = None,
    audio_format: str | None = None,
) -> Path:
    client = get_groq_client()
    model_name = model or os.getenv("GROQ_TTS_MODEL", GROQ_TTS_DEFAULT_MODEL)
    voice_name = voice or os.getenv("GROQ_TTS_VOICE", GROQ_TTS_DEFAULT_VOICE)
    response_format = audio_format or os.getenv("GROQ_TTS_FORMAT", GROQ_TTS_DEFAULT_FORMAT)

    if response_format and output_path.suffix.lower() != f".{response_format}":
        output_path = output_path.with_suffix(f".{response_format}")

    response = client.audio.speech.create(
        model=model_name,
        voice=voice_name,
        input=text,
        response_format=response_format,
    )
    response.write_to_file(str(output_path))
    return output_path


def synthesize_speech(
    text: str,
    output_path: str | Path,
    *,
    model: str | None = None,
    voice: str | None = None,
    audio_format: str | None = None,
) -> Path:
    output_path = Path(output_path)

    deepgram_exc: Exception | None = None
    tts_debug = os.getenv("TTS_DEBUG") == "1"
    try:
        return _synthesize_deepgram(text, output_path, model=model)
    except Exception as exc:
        deepgram_exc = exc
        if tts_debug:
            print(f"TTS debug: Deepgram error: {exc!r}")

    try:
        return _synthesize_groq(
            text,
            output_path,
            model=model,
            voice=voice,
            audio_format=audio_format,
        )
    except Exception as exc:
        if tts_debug:
            print(f"TTS debug: Groq error: {exc!r}")
        if deepgram_exc and _is_rate_limit_error(deepgram_exc):
            raise RuntimeError("Deepgram TTS rate-limited and Groq TTS failed.") from exc
        raise RuntimeError("Deepgram TTS failed and Groq TTS fallback failed.") from exc
