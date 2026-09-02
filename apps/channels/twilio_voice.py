"""
apps/channels/twilio_voice.py
------------------------------
E5: Twilio Voice channel adapter — Media Streams over WebSocket.

Call flow:
  1. Inbound call → Twilio hits POST /channels/twilio/voice/incoming
  2. We return TwiML with <Connect><Stream> pointing to /channels/twilio/voice/stream
  3. Twilio opens WS to /channels/twilio/voice/stream
  4. Twilio sends μ-law 8 kHz audio frames as base64 JSON
  5. Adapter: accumulate audio → STT (Groq Whisper) → LLM pipeline → TTS (Deepgram)
  6. TTS audio is encoded as μ-law 8 kHz and sent back to Twilio

Audio format:
  Twilio Media Streams: PCMU (G.711 μ-law), 8 000 Hz, mono, 20 ms frames
  STT input: we accumulate frames until VAD silence, then decode to PCM, WAV-wrap
  TTS output: Deepgram returns linear16 → we transcode to μ-law via audioop

VAD strategy (simple energy-based):
  We accumulate audio until 800 ms of silence, then trigger STT.
  Full Silero VAD (E2) is deferred — this gives a functional baseline.

Required env vars (same Twilio account as WhatsApp):
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_VOICE_NUMBER
  DEEPGRAM_API_KEY      (for TTS)
  GROQ_API_KEY          (for STT via Whisper)
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import struct
import time
import wave
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect

from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
from core.config import settings
from core.identity import identity_manager
from core.output_clean import clean_text_for_tts

router = APIRouter(prefix="/channels/twilio/voice", tags=["twilio_voice"])

# Twilio Media Streams constants
_SAMPLE_RATE = 8000   # Hz — fixed by Twilio PCMU
_FRAME_MS = 20        # ms per frame
_FRAME_SAMPLES = int(_SAMPLE_RATE * _FRAME_MS / 1000)  # 160 samples
_SILENCE_THRESHOLD_ENERGY = 300  # μ-law RMS threshold for silence detection
_SILENCE_TRIGGER_MS = 800         # ms of silence before STT fires
_SILENCE_FRAMES = int(_SILENCE_TRIGGER_MS / _FRAME_MS)  # 40 frames


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """Decode μ-law bytes to signed 16-bit PCM."""
    return audioop.ulaw2lin(ulaw_bytes, 2)


def _pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """Encode signed 16-bit PCM to μ-law bytes."""
    return audioop.lin2ulaw(pcm_bytes, 2)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _frame_energy(ulaw_bytes: bytes) -> float:
    """Estimate energy of a μ-law frame (RMS of decoded PCM)."""
    pcm = _ulaw_to_pcm16(ulaw_bytes)
    samples = struct.unpack(f"{len(pcm)//2}h", pcm)
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


# ---------------------------------------------------------------------------
# STT — Groq Whisper
# ---------------------------------------------------------------------------

async def _transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribe WAV bytes using Groq Whisper."""
    api_key = (
        settings.groq_api_key.get_secret_value() if settings.groq_api_key
        else settings.groq_api_key2.get_secret_value() if settings.groq_api_key2
        else None
    )
    if not api_key:
        return ""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": "whisper-large-v3-turbo", "response_format": "text"},
            timeout=15.0,
        )
    if resp.status_code != 200:
        print(f"[TwilioVoice] STT failed: {resp.status_code}")
        return ""
    return resp.text.strip()


# ---------------------------------------------------------------------------
# TTS — Deepgram Aura (returns linear16, we transcode to μ-law)
# ---------------------------------------------------------------------------

async def _synthesize_ulaw(text: str) -> bytes:
    """Synthesize text → μ-law 8 kHz audio via Deepgram."""
    api_key = settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else None
    if not api_key:
        return b""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.deepgram.com/v1/speak",
            params={
                "model": "aura-2-orion-en",
                "encoding": "linear16",
                "sample_rate": str(_SAMPLE_RATE),
                "container": "none",
            },
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            json={"text": text},
            timeout=20.0,
        )
    if resp.status_code != 200:
        print(f"[TwilioVoice] TTS failed: {resp.status_code}")
        return b""

    pcm_bytes = resp.content
    return _pcm16_to_ulaw(pcm_bytes)


# ---------------------------------------------------------------------------
# TwiML entry-point
# ---------------------------------------------------------------------------

@router.post("/incoming")
async def voice_incoming(request: Request):
    """
    Twilio calls this when a voice call arrives.
    Returns TwiML that tells Twilio to open a Media Stream to /stream.
    """
    host = request.headers.get("host") or request.base_url.netloc
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/channels/twilio/voice/stream"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}" />'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Media Streams WebSocket handler
# ---------------------------------------------------------------------------

@router.websocket("/stream")
async def voice_stream(ws: WebSocket):
    """
    Twilio Media Streams WebSocket.

    Message types we handle:
      connected   — handshake, ignore
      start       — stream started, extract call_sid / from number
      media       — audio payload (base64 PCMU)
      stop        — call ended
    """
    await ws.accept()

    call_sid: str = ""
    from_number: str = ""
    user_id: Optional[str] = None
    audio_buffer: list[bytes] = []   # accumulated μ-law frames
    silence_count: int = 0
    speaking: bool = False

    async def _flush_and_respond() -> None:
        """STT the buffer, dispatch to Turtle, send TTS reply."""
        nonlocal audio_buffer, silence_count, speaking
        if not audio_buffer:
            return

        pcm_frames = b"".join(_ulaw_to_pcm16(f) for f in audio_buffer)
        audio_buffer = []
        silence_count = 0
        speaking = False

        wav_bytes = _pcm_to_wav(pcm_frames)
        text = await _transcribe_audio(wav_bytes)
        if not text:
            return

        print(f"[TwilioVoice] STT: {text!r}")

        # A caller whose number Twilio didn't share must NOT collapse onto a
        # shared "anon" identity — the dispatch pipeline now builds a full
        # memory-bearing state per user_id, so a shared id would pool every
        # anonymous caller's session history and memory into one account.
        # Scope the fallback to this call instead.
        uid = user_id or f"anon_{call_sid or 'unknown'}"
        event = TurtleEvent(
            user_id=uid,
            channel="twilio_voice",
            modality="voice",
            content=text,
            message_id=call_sid,
        )
        response: TurtleResponse = await dispatch_event(event)
        reply_text = clean_text_for_tts(response.content)

        ulaw_audio = await _synthesize_ulaw(reply_text)
        if not ulaw_audio:
            return

        # Stream audio back to Twilio in PCMU chunks
        chunk_size = _FRAME_SAMPLES  # 160 bytes per frame
        for i in range(0, len(ulaw_audio), chunk_size):
            chunk = ulaw_audio[i: i + chunk_size]
            payload = base64.b64encode(chunk).decode()
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": call_sid,
                "media": {"payload": payload},
            }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event_name = msg.get("event", "")

            if event_name == "connected":
                continue

            elif event_name == "start":
                start_data = msg.get("start", {})
                call_sid = start_data.get("callSid", "")
                custom = start_data.get("customParameters", {})
                from_number = custom.get("from", "")
                if from_number:
                    user_id = await identity_manager.resolve_user("twilio_voice", from_number)
                print(f"[TwilioVoice] Stream started: call_sid={call_sid} from={from_number}")

            elif event_name == "media":
                payload_b64 = msg.get("media", {}).get("payload", "")
                if not payload_b64:
                    continue
                ulaw_frame = base64.b64decode(payload_b64)
                energy = _frame_energy(ulaw_frame)

                if energy > _SILENCE_THRESHOLD_ENERGY:
                    speaking = True
                    silence_count = 0
                    audio_buffer.append(ulaw_frame)
                elif speaking:
                    silence_count += 1
                    audio_buffer.append(ulaw_frame)
                    if silence_count >= _SILENCE_FRAMES:
                        # Enough silence — fire STT
                        asyncio.create_task(_flush_and_respond())

            elif event_name == "stop":
                print(f"[TwilioVoice] Stream stopped: {call_sid}")
                # Flush any remaining audio
                if audio_buffer:
                    await _flush_and_respond()
                break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[TwilioVoice] Stream error: {exc}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
