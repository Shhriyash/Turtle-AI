"""
core/streaming_tts.py
---------------------
E3: Streaming TTS with sentence-boundary chunking.

Instead of synthesising the full LLM output as one monolithic WAV (the current
approach in openrouter_tts.py:synthesize_speech), this module:
  1. Receives LLM output tokens as they stream.
  2. Groups tokens into sentences at sentence boundaries (. ! ? \\n\\n).
  3. Fires each sentence to Deepgram TTS in parallel as soon as the sentence
     boundary is detected — while the next sentence is still generating.
  4. Yields (sentence_text, audio_bytes) tuples back to the caller.

E1/E2 (streaming STT + VAD barge-in) are deferred — marked as TODO below.
E4 latency budget: TTS_FIRST_BYTE_MAX_MS=600 ms soft target, handled by the
     caller's asyncio.wait_for at the handler level.

Usage::

    async for sentence, audio_bytes in stream_tts_sentences(text_generator()):
        await ws.send_bytes(audio_bytes)   # stream chunk to client
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import threading
import time
from typing import AsyncIterator, Callable, Iterator


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------

# Sentence ends at: period/!/?/… followed by whitespace or end-of-string,
# OR double-newline (paragraph break).
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?…])\s+|(?<=\n)\n+'
)

# Minimum sentence length to fire TTS (avoid triggering on "ok." alone)
_MIN_SENTENCE_CHARS = 12


def split_into_sentences(text: str) -> list[str]:
    """Split text into TTS-ready sentence chunks."""
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text).strip()
    if not text:
        return []

    parts = _SENTENCE_BOUNDARY.split(text)
    sentences: list[str] = []
    buffer = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        buffer = (buffer + " " + part).strip() if buffer else part
        # Fire when we have a real sentence (ends with sentence-ending punctuation)
        if re.search(r'[.!?…]$', buffer) and len(buffer) >= _MIN_SENTENCE_CHARS:
            sentences.append(buffer)
            buffer = ""

    # Flush any remaining buffer (last sentence without trailing punctuation)
    if buffer and len(buffer) >= 3:
        sentences.append(buffer)

    return sentences


class SentenceAccumulator:
    """Stream tokens in, get complete sentences out."""

    def __init__(self) -> None:
        self._buffer = ""
        self._sentences: list[str] = []

    def feed(self, token: str) -> list[str]:
        """Feed a token fragment; return any newly completed sentences."""
        self._buffer += token
        completed: list[str] = []

        # Check for sentence boundary in accumulated buffer
        while True:
            m = _SENTENCE_BOUNDARY.search(self._buffer)
            if not m:
                break
            sentence = self._buffer[: m.start()].strip()
            remainder = self._buffer[m.end() :]
            if sentence and len(sentence) >= _MIN_SENTENCE_CHARS:
                completed.append(sentence)
            self._buffer = remainder

        return completed

    def flush(self) -> list[str]:
        """Return any remaining buffered text as the final sentence."""
        buf = self._buffer.strip()
        self._buffer = ""
        if buf and len(buf) >= 3:
            return [buf]
        return []


# ---------------------------------------------------------------------------
# Per-sentence Deepgram TTS synthesis (async)
# ---------------------------------------------------------------------------

async def _synthesize_sentence_async(
    text: str,
    *,
    model: str | None = None,
    speed: float | None = None,
) -> bytes:
    """Synthesise one sentence and return a complete WAV as bytes.

    Delegates to ``synthesize_speech_bytes`` (Deepgram REST primary, Groq
    fallback, WS opt-in), run in a thread executor so the blocking provider call
    never stalls the event loop. Each returned blob is a self-describing RIFF/WAV
    the browser can hand straight to ``decodeAudioData`` — no temp file, and none
    of the WebSocket idle-drain wait that used to gate first audio.
    """
    from functools import partial

    from core.openrouter_tts import synthesize_speech_bytes

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(synthesize_speech_bytes, text, model=model, speed=speed),
    )


# ---------------------------------------------------------------------------
# Streaming TTS entry points
# ---------------------------------------------------------------------------

async def stream_tts_from_text(
    full_text: str,
    *,
    model: str | None = None,
    speed: float | None = None,
    tts_timeout_s: float = 8.0,
) -> AsyncIterator[tuple[str, bytes]]:
    """E3: Split full_text into sentences and yield (sentence, audio_bytes) per sentence.

    First audio chunk should arrive within ~600 ms of TTS start (Deepgram
    aura-2-orion-en p50 is ~350-450 ms per sentence).

    Args:
        full_text: The complete LLM output text.
        model: Deepgram model override.
        speed: TTS speed (0.7–1.5).
        tts_timeout_s: Per-sentence synthesis timeout.

    Yields:
        (sentence_text, wav_bytes) tuples in order.
    """
    sentences = split_into_sentences(full_text)
    if not sentences:
        return

    # Fire sentences concurrently via a task queue; yield in order.
    tasks: list[asyncio.Task] = []
    for sentence in sentences:
        task = asyncio.create_task(
            _synthesize_sentence_async(sentence, model=model, speed=speed)
        )
        tasks.append((sentence, task))

    for sentence, task in tasks:
        try:
            audio_bytes = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=tts_timeout_s,
            )
            yield sentence, audio_bytes
        except asyncio.TimeoutError:
            print(f"LOG: TTS sentence timeout ({tts_timeout_s:.0f} s) for: {sentence[:40]!r}")
            task.cancel()
            continue
        except Exception as exc:
            print(f"LOG: TTS sentence error for {sentence[:40]!r}: {exc}")
            task.cancel()
            continue


async def stream_tts_from_token_stream(
    token_iterator: AsyncIterator[str],
    *,
    model: str | None = None,
    speed: float | None = None,
    tts_timeout_s: float = 8.0,
    clean_fn: "Callable[[str], str] | None" = None,
) -> AsyncIterator[tuple[str, bytes]]:
    """E3: Sentence-boundary chunked TTS from a streaming token iterator.

    As LLM tokens arrive, groups them into sentences.  Fires TTS for each
    completed sentence while the next sentence is still generating.

    E1/E2 TODO: When STT streaming + barge-in is implemented, the caller will
    cancel this iterator via an asyncio.CancelledError when VAD detects speech.

    Args:
        token_iterator: Async iterator of LLM token strings.
        model: Deepgram model override.
        speed: TTS speed.
        tts_timeout_s: Per-sentence synthesis timeout.
        clean_fn: optional per-sentence normaliser (e.g. strip markdown for TTS)
            applied before synthesis; a sentence that cleans to empty is skipped.

    Yields:
        (spoken_sentence_text, wav_bytes) tuples as they complete, in order.
    """
    accumulator = SentenceAccumulator()
    # Ordered queue: sentences synthesise concurrently (tasks fire the moment a
    # boundary is detected) but MUST be yielded strictly in sentence order, or
    # playback plays them scrambled. We only ever release from the front.
    pending_tasks: list[tuple[str, asyncio.Task]] = []

    def _spawn(raw_sentence: str) -> None:
        spoken = clean_fn(raw_sentence) if clean_fn else raw_sentence
        if not spoken or not spoken.strip():
            return
        task = asyncio.create_task(
            _synthesize_sentence_async(spoken, model=model, speed=speed)
        )
        pending_tasks.append((spoken, task))

    async for token in token_iterator:
        for sentence in accumulator.feed(token):
            _spawn(sentence)

        # Non-blocking release of any FRONT tasks already finished — never skip
        # ahead to a later sentence that happens to finish first.
        while pending_tasks and pending_tasks[0][1].done():
            sentence, task = pending_tasks.pop(0)
            try:
                yield sentence, task.result()
            except Exception as exc:
                print(f"LOG: TTS sentence error: {exc}")

    # Flush the final partial sentence after the token stream ends.
    for sentence in accumulator.flush():
        _spawn(sentence)

    # Drain the rest in strict order, awaiting each front task in turn.
    for sentence, task in pending_tasks:
        try:
            audio_bytes = await asyncio.wait_for(asyncio.shield(task), timeout=tts_timeout_s)
            yield sentence, audio_bytes
        except asyncio.TimeoutError:
            print(f"LOG: TTS sentence timeout for: {sentence[:40]!r}")
            task.cancel()
        except Exception as exc:
            print(f"LOG: TTS sentence error: {exc}")
            task.cancel()


# ---------------------------------------------------------------------------
# E1/E2 stubs (deferred)
# ---------------------------------------------------------------------------

# TODO E1: Implement streaming STT via Deepgram streaming or Groq Whisper streaming.
#   New file: core/stt_streaming.py exposing AsyncIterator[str] of partial transcripts.
#   Frontend receives {type: "transcription_partial", text} frames over the WS.

# TODO E2: Wire rtc_vad/vad_fastrtc.py (Silero VAD) into WebSocket audio path.
#   On VAD speech-start while TTS is streaming, cancel the stream_tts_from_token_stream
#   iterator and send {type: "barge_in"} to the client.
#   New orchestration file: core/voice_pipeline.py.
