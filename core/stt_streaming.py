"""
core/stt_streaming.py
---------------------
Phase 1 (E1): streaming STT with interim transcripts and model-based end-of-turn.

Wraps Deepgram Flux (v2 listen) — a conversational STT model that emits turn
events (StartOfTurn / EagerEndOfTurn / EndOfTurn) with ~260ms median end-of-turn
detection, replacing the fixed client-side silence timer. Transcription is
effectively done the moment the user stops talking, and interim hypotheses stream
back for live captions.

The Deepgram SDK exposes a *synchronous*, thread-based websocket. To bridge it to
the server's asyncio world without blocking the event loop, one session runs two
worker threads over a single connection (concurrent send + recv is supported):
  - a RECV thread drains ``conn.recv()`` and hands each event to the event loop
    via ``call_soon_threadsafe``;
  - a SEND thread drains an audio queue and calls ``conn.send_media``.

Usage::

    stt = FluxStreamingSTT(sample_rate=16000)
    await stt.start()
    async def pump():
        async for ev in stt.events():
            if ev.kind == "update":       # interim transcript (live caption)
                ...
            elif ev.kind == "end_of_turn":  # final transcript -> run the turn
                run_turn(ev.transcript)
    # feed audio frames as they arrive from the client:
    await stt.send_audio(pcm_frame_bytes)
    ...
    await stt.finish()   # flush; server emits the final EndOfTurn then closes
    await stt.aclose()
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


# Sentinels for the send queue.
_CLOSE = object()   # ask Deepgram to finalise the current turn and close
_STOP = object()    # tear the sender thread down


@dataclass
class TurnEvent:
    """One event from the streaming STT session.

    kind:
      - "connected"        handshake complete
      - "update"           interim transcript (may be partial / empty)
      - "start_of_turn"    the user started speaking
      - "eager_end_of_turn" speculative end-of-turn (low-latency, may resume)
      - "turn_resumed"     a prior eager end-of-turn was retracted
      - "end_of_turn"      the user finished; ``transcript`` is final for the turn
      - "error"            fatal error from the provider
      - "closed"           the session is over; no more events follow
    """
    kind: str
    transcript: str = ""
    raw: Any = None


# Deepgram Flux event names -> our normalised kinds.
_EVENT_KIND = {
    "Update": "update",
    "StartOfTurn": "start_of_turn",
    "EagerEndOfTurn": "eager_end_of_turn",
    "TurnResumed": "turn_resumed",
    "EndOfTurn": "end_of_turn",
}


class FluxStreamingSTT:
    """A single Deepgram Flux streaming-STT session bridged to asyncio."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        sample_rate: int = 16000,
        encoding: str = "linear16",
        eot_threshold: Optional[float] = None,
        eager_eot_threshold: Optional[float] = None,
        eot_timeout_ms: Optional[int] = None,
        keyterms: Optional[list] = None,
    ) -> None:
        self.model = model or os.getenv("DEEPGRAM_STT_STREAM_MODEL", "flux-general-en")
        self.sample_rate = int(sample_rate)
        self.encoding = encoding
        self.eot_threshold = eot_threshold
        self.eager_eot_threshold = eager_eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        # Bias recognition toward these terms (the user's name, emails, contacts)
        # so Flux stops mangling proper nouns it hasn't heard before.
        self.keyterms = [k for k in (keyterms or []) if k and str(k).strip()]

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_q: "asyncio.Queue[TurnEvent]" = asyncio.Queue()
        self._send_q: "queue.Queue[Any]" = queue.Queue()
        self._conn: Any = None
        self._conn_ready = threading.Event()
        self._conn_error: Optional[Exception] = None
        self._stop = threading.Event()
        self._session_thread: Optional[threading.Thread] = None
        self._closed_emitted = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self, *, connect_timeout: float = 8.0) -> None:
        """Open the connection and spawn the session worker.

        A single owning thread creates the socket, sends audio, and hosts an
        inner recv sub-thread — the sync websocket client misbehaves if audio is
        sent from a thread other than the one that opened it, so send and connect
        must share a thread.
        """
        self._loop = asyncio.get_event_loop()
        self._session_thread = threading.Thread(
            target=self._session_worker, name="flux-stt", daemon=True
        )
        self._session_thread.start()

        # Wait (off the event loop) until the socket is connected or errored.
        ready = await self._loop.run_in_executor(
            None, self._conn_ready.wait, connect_timeout
        )
        if not ready:
            self._stop.set()
            raise RuntimeError("Flux STT connect timed out")
        if self._conn_error is not None:
            raise RuntimeError(f"Flux STT connect failed: {self._conn_error}")

    async def send_audio(self, pcm: bytes) -> None:
        """Queue a PCM frame for transmission (non-blocking)."""
        if pcm:
            self._send_q.put(pcm)

    async def finish(self) -> None:
        """Signal end of audio; Deepgram emits the final EndOfTurn, then closes."""
        self._send_q.put(_CLOSE)

    async def aclose(self) -> None:
        """Tear the session down and stop both worker threads."""
        self._stop.set()
        self._send_q.put(_STOP)
        conn = self._conn
        if conn is not None:
            # Best-effort: closing the stream unblocks the recv() loop.
            try:
                conn.send_close_stream()
            except Exception:
                pass

    def events(self) -> "AsyncIterator[TurnEvent]":
        """Async-iterate normalised turn events until the session closes."""
        return self._event_iter()

    async def _event_iter(self) -> "AsyncIterator[TurnEvent]":
        while True:
            ev = await self._event_q.get()
            yield ev
            if ev.kind in ("closed", "error"):
                break

    # -- worker threads -----------------------------------------------------

    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
        }
        if self.eot_threshold is not None:
            kwargs["eot_threshold"] = f"{self.eot_threshold:.2f}"
        if self.eager_eot_threshold is not None:
            kwargs["eager_eot_threshold"] = f"{self.eager_eot_threshold:.2f}"
        if self.eot_timeout_ms is not None:
            kwargs["eot_timeout_ms"] = str(int(self.eot_timeout_ms))
        if self.keyterms:
            # Flux accepts a repeatable keyterm parameter (str or sequence).
            kwargs["keyterm"] = list(self.keyterms)
        return kwargs

    def _emit(self, ev: TurnEvent) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._event_q.put_nowait, ev)
        except RuntimeError:
            # Event loop is gone (shutdown race) — nothing to deliver to.
            pass

    def _session_worker(self) -> None:
        """Owning thread: connect, host a recv sub-thread, and send audio."""
        from tools.tts.client import get_deepgram_client

        try:
            client = get_deepgram_client()
            with client.listen.v2.connect(**self._connect_kwargs()) as conn:
                self._conn = conn
                recv_thread = threading.Thread(
                    target=self._recv_loop, args=(conn,),
                    name="flux-stt-recv", daemon=True,
                )
                recv_thread.start()
                self._conn_ready.set()

                # Drain the audio queue from the SAME thread that opened the
                # socket. _CLOSE finalises the current turn; _STOP tears down.
                debug = os.getenv("STT_STREAM_DEBUG") == "1"
                sent = 0
                while not self._stop.is_set():
                    try:
                        item = self._send_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is _STOP:
                        break
                    if item is _CLOSE:
                        try:
                            conn.send_close_stream()
                        except Exception as exc:
                            if debug:
                                print(f"STT debug: send_close_stream failed: {exc!r}")
                        continue
                    try:
                        conn.send_media(item)
                        sent += 1
                    except Exception as exc:
                        if debug:
                            print(f"STT debug: send_media failed after {sent} frames: {exc!r}")
                        break

                if debug:
                    print(f"STT debug: send loop ended after {sent} frames")
                # recv() stays blocked until this `with` exits and closes the
                # socket, so a long join only delays that close. Keep it short:
                # the recv thread is a daemon and ends the moment the socket shuts.
                recv_thread.join(timeout=0.2)
        except Exception as exc:
            self._conn_error = exc
            self._conn_ready.set()  # unblock start() even on failure
            self._emit(TurnEvent(kind="error", raw=exc))
        finally:
            if not self._closed_emitted:
                self._closed_emitted = True
                self._emit(TurnEvent(kind="closed"))

    def _recv_loop(self, conn: Any) -> None:
        while not self._stop.is_set():
            try:
                m = conn.recv()
            except Exception:
                break
            if m is None:
                break
            self._emit(self._normalise(m))

    @staticmethod
    def _normalise(message: Any) -> TurnEvent:
        cls_name = type(message).__name__
        if cls_name == "ListenV2Connected":
            return TurnEvent(kind="connected", raw=message)
        if cls_name == "ListenV2FatalError":
            return TurnEvent(kind="error", raw=message)
        # ListenV2TurnInfo — carries .event and .transcript.
        event_name = getattr(message, "event", None) or getattr(message, "type", None)
        transcript = getattr(message, "transcript", None) or ""
        kind = _EVENT_KIND.get(event_name, "update")
        return TurnEvent(kind=kind, transcript=transcript, raw=message)
