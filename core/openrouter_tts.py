from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from tools.tts.client import get_deepgram_client, get_groq_client

DEEPGRAM_TTS_DEFAULT_MODEL = "aura-2-orion-en"
DEEPGRAM_TTS_DEFAULT_ENCODING = "linear16"
DEEPGRAM_TTS_DEFAULT_CONTAINER = "wav"
DEEPGRAM_TTS_DEFAULT_SAMPLE_RATE = "24000"
GROQ_TTS_DEFAULT_MODEL = "canopylabs/orpheus-v1-english"
GROQ_TTS_DEFAULT_VOICE = "orion"
GROQ_TTS_DEFAULT_FORMAT = "wav"


def _coerce_speed(value: float | None) -> float:
    if value is None:
        raw = os.getenv("TURTLE_TTS_SPEED") or os.getenv("DEEPGRAM_TTS_SPEED") or "1.2"
        try:
            parsed = float(raw)
        except Exception:
            parsed = 1.2
    else:
        parsed = float(value)
    return max(0.7, min(1.5, parsed))

def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return any(token in message for token in ["rate limit", "rate_limit", "quota", "insufficient"])


def _is_unsupported_speed_error(exc: Exception) -> bool:
    # SDK exceptions may hide server payloads in repr/body/headers instead of str().
    parts: list[str] = [str(exc), repr(exc)]

    body = getattr(exc, "body", None)
    if body is not None:
        try:
            if isinstance(body, (bytes, bytearray)):
                parts.append(body.decode("utf-8", errors="ignore"))
            else:
                parts.append(str(body))
        except Exception:
            pass

    headers = getattr(exc, "headers", None)
    if headers is not None:
        try:
            parts.append(str(headers))
            dg_error = headers.get("dg-error") if hasattr(headers, "get") else None
            if dg_error:
                parts.append(str(dg_error))
        except Exception:
            pass

    message = " ".join(parts).lower()
    if "speed" not in message:
        return False
    return (
        "does not support" in message
        or "requested model does not support" in message
        or "speed_out_of_range" in message
        or "unsupported" in message
    )


def _synthesize_deepgram_ws(
    text: str,
    output_path: Path,
    *,
    model: str | None = None,
    speed: float | None = None,
) -> Path:
    """Synthesize speech using Deepgram TTS WebSocket streaming API."""
    client = get_deepgram_client()
    if not text or not text.strip():
        raise RuntimeError("TTS text is empty.")

    model_name = model or os.getenv("DEEPGRAM_TTS_MODEL", DEEPGRAM_TTS_DEFAULT_MODEL)
    encoding = os.getenv("DEEPGRAM_TTS_ENCODING", DEEPGRAM_TTS_DEFAULT_ENCODING)
    container = os.getenv("DEEPGRAM_TTS_CONTAINER", DEEPGRAM_TTS_DEFAULT_CONTAINER)
    sample_rate = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", DEEPGRAM_TTS_DEFAULT_SAMPLE_RATE))
    speed_value = _coerce_speed(speed)

    if output_path.suffix.lower() != f".{container}":
        output_path = output_path.with_suffix(f".{container}")

    try:
        from deepgram.core.events import EventType
        from deepgram.speak.v1.types import SpeakV1Text
    except Exception as exc:
        raise RuntimeError("Deepgram WS TTS types unavailable in current SDK.") from exc

    base_connect_kwargs = {
        "model": model_name,
        "encoding": encoding,
        "sample_rate": sample_rate,
    }

    def _stream_ws(use_speed: bool) -> Path:
        connect_kwargs = dict(base_connect_kwargs)
        if use_speed:
            # Speed controls are exposed as query parameters in Deepgram docs.
            try:
                from deepgram.core.request_options import RequestOptions

                connect_kwargs["request_options"] = RequestOptions(
                    additional_query_parameters={"speed": f"{speed_value:.2f}"},
                )
            except Exception:
                # Keep WS working even when request_options isn't present in installed SDK.
                pass

        audio_chunks: list[bytes] = []
        chunks_lock = threading.Lock()
        error_holder: list[Exception] = []
        stream_closed = threading.Event()
        flush_sent_at = 0.0

        idle_drain_ms_raw = os.getenv("TTS_WS_IDLE_DRAIN_MS", "900")
        try:
            idle_drain_ms = max(300, min(3000, int(idle_drain_ms_raw)))
        except Exception:
            idle_drain_ms = 900
        idle_drain_s = idle_drain_ms / 1000.0

        with client.speak.v1.connect(**connect_kwargs) as dg_connection:
            def on_message(message) -> None:
                if isinstance(message, (bytes, bytearray)):
                    with chunks_lock:
                        audio_chunks.append(bytes(message))

            def on_close(_event=None) -> None:
                stream_closed.set()

            def on_error(error) -> None:
                if isinstance(error, Exception):
                    error_holder.append(error)
                else:
                    error_holder.append(RuntimeError(str(error)))

            dg_connection.on(EventType.MESSAGE, on_message)
            dg_connection.on(EventType.ERROR, on_error)
            try:
                dg_connection.on(EventType.CLOSE, on_close)
            except Exception:
                # Some SDK versions may not expose CLOSE event.
                pass
            dg_connection.start_listening()
            dg_connection.send_text(SpeakV1Text(text=text))
            dg_connection.send_flush()
            flush_sent_at = time.time()

            # Wait until close signal, or a conservative idle drain window after flush.
            deadline = time.time() + 20.0
            last_chunk_count = -1
            last_change = time.time()
            while time.time() < deadline:
                with chunks_lock:
                    chunk_count = len(audio_chunks)
                if chunk_count != last_chunk_count:
                    last_chunk_count = chunk_count
                    last_change = time.time()
                idle_for = time.time() - last_change
                age_since_flush = time.time() - flush_sent_at

                if stream_closed.is_set() and chunk_count > 0 and idle_for >= 0.05:
                    break

                # Fallback drain path when CLOSE event isn't surfaced by SDK.
                if chunk_count > 0 and idle_for >= idle_drain_s and age_since_flush >= 0.9:
                    break
                time.sleep(0.05)

            dg_connection.send_close()

        if error_holder and not audio_chunks:
            raise RuntimeError(f"Deepgram WS TTS error: {error_holder[0]}")
        if not audio_chunks:
            raise RuntimeError("Deepgram WS TTS produced no audio data.")

        output_path.write_bytes(b"".join(audio_chunks))
        return output_path

    try:
        return _stream_ws(use_speed=True)
    except Exception as exc:
        # Some models/accounts reject speed controls. Retry WS immediately without speed.
        if _is_unsupported_speed_error(exc):
            return _stream_ws(use_speed=False)
        raise


def _synthesize_deepgram_rest(
    text: str,
    output_path: Path,
    *,
    model: str | None = None,
    speed: float | None = None,
) -> Path:
    """Fallback Deepgram REST synthesis path."""
    client = get_deepgram_client()
    model_name = model or os.getenv("DEEPGRAM_TTS_MODEL", DEEPGRAM_TTS_DEFAULT_MODEL)
    encoding = os.getenv("DEEPGRAM_TTS_ENCODING", DEEPGRAM_TTS_DEFAULT_ENCODING)
    container = os.getenv("DEEPGRAM_TTS_CONTAINER", DEEPGRAM_TTS_DEFAULT_CONTAINER)
    sample_rate = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", DEEPGRAM_TTS_DEFAULT_SAMPLE_RATE))
    speed_value = _coerce_speed(speed)

    if output_path.suffix.lower() != f".{container}":
        output_path = output_path.with_suffix(f".{container}")

    request_kwargs = {
        "text": text,
        "model": model_name,
        "encoding": encoding,
        "container": container,
        "sample_rate": sample_rate,
    }
    try:
        from deepgram.core.request_options import RequestOptions

        request_kwargs["request_options"] = RequestOptions(
            additional_query_parameters={"speed": f"{speed_value:.2f}"},
        )
    except Exception:
        pass

    response = client.speak.v1.audio.generate(
        **request_kwargs,
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
    speed: float | None = None,
) -> Path:
    output_path = Path(output_path)

    deepgram_exc: Exception | None = None
    tts_debug = os.getenv("TTS_DEBUG") == "1"
    try:
        return _synthesize_deepgram_ws(text, output_path, model=model, speed=speed)
    except Exception as exc:
        deepgram_exc = exc
        if tts_debug:
            print(f"TTS debug: Deepgram WS error: {exc!r}")

    # Safety fallback for environments without WS support.
    try:
        return _synthesize_deepgram_rest(text, output_path, model=model, speed=speed)
    except Exception as exc:
        deepgram_exc = exc
        if tts_debug:
            print(f"TTS debug: Deepgram REST fallback error: {exc!r}")

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
