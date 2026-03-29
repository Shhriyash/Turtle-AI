import time
import os
import queue
import threading
import numpy as np
import sounddevice as sd

from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text

from tools.tts.client import get_deepgram_client

TTS_TEXT = "Hello, this is a streaming text to speech example using Deepgram."
MODEL = "aura-2-apollo-en"
SAMPLE_RATE = 48000
ENCODING = "linear16"


def stream_tts(
    text: str,
    *,
    model: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    encoding: str = ENCODING,
    idle_timeout: float = 1.0,
    start_timeout: float = 3.0,
    max_total_time: float | None = 8.0,
    debug: bool = False,
) -> bool:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    done_event = threading.Event()
    started_audio = threading.Event()
    last_audio_time = {"t": 0.0}

    def _enqueue_audio(raw: bytes) -> None:
        if not raw:
            return
        array = np.frombuffer(raw, dtype=np.int16)
        audio_queue.put(array)
        last_audio_time["t"] = time.time()
        if not started_audio.is_set():
            print(f"Received audio chunk ({len(raw)} bytes)")
        started_audio.set()

    def on_message(message) -> None:
        if isinstance(message, bytes):
            _enqueue_audio(message)
            return

        msg_type = getattr(message, "type", "Unknown")
        print(f"Received {msg_type} event")

        data = None
        if hasattr(message, "data"):
            data = message.data
        elif isinstance(message, dict):
            data = message.get("data")

        if isinstance(data, (bytes, bytearray)):
            _enqueue_audio(bytes(data))
            return
        if isinstance(data, str):
            try:
                import base64
                decoded = base64.b64decode(data)
                _enqueue_audio(decoded)
            except Exception:
                pass

        if debug:
            try:
                payload = message if isinstance(message, dict) else vars(message)
                print(f"Debug message payload: {payload}")
            except Exception:
                print(f"Debug message repr: {message!r}")

    def audio_callback(outdata, frames, _time, status):
        if status:
            pass
        samples = np.zeros(frames, dtype=np.int16)
        filled = 0
        while filled < frames:
            try:
                chunk = audio_queue.get_nowait()
            except queue.Empty:
                break
            chunk_len = len(chunk)
            remaining = frames - filled
            if chunk_len <= remaining:
                samples[filled:filled + chunk_len] = chunk
                filled += chunk_len
            else:
                samples[filled:] = chunk[:remaining]
                audio_queue.put(chunk[remaining:])
                filled = frames
        outdata[:, 0] = samples

    def idle_monitor(start_time: float):
        while not done_event.is_set():
            if max_total_time is not None and time.time() - start_time >= max_total_time:
                print("Max TTS streaming time reached.")
                done_event.set()
                break
            if not started_audio.is_set():
                if time.time() - start_time >= start_timeout:
                    print("No audio received from Deepgram within timeout.")
                    done_event.set()
                    break
            if started_audio.is_set() and audio_queue.empty():
                if time.time() - last_audio_time["t"] >= idle_timeout:
                    done_event.set()
                    break
            time.sleep(0.05)

    try:
        deepgram = get_deepgram_client()
        model_name = model or MODEL

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=audio_callback,
        ):
            with deepgram.speak.v1.connect(
                model=model_name,
                encoding=encoding,
                sample_rate=sample_rate,
            ) as dg_connection:
                dg_connection.on(EventType.OPEN, lambda _: print("Connection opened"))
                dg_connection.on(EventType.MESSAGE, on_message)
                dg_connection.on(EventType.CLOSE, lambda _: done_event.set())
                dg_connection.on(EventType.ERROR, lambda error: print(f"Error: {error}"))

                dg_connection.start_listening()
                start_time = time.time()
                dg_connection.send_text(SpeakV1Text(text=text))
                dg_connection.send_flush()
                last_audio_time["t"] = start_time

                monitor = threading.Thread(target=idle_monitor, args=(start_time,), daemon=True)
                monitor.start()

                while not done_event.is_set():
                    time.sleep(0.05)

                dg_connection.send_close()

        return started_audio.is_set()
    except Exception as e:
        print(f"Streaming TTS error: {e}")
        return False


def main():
    debug = os.getenv("DEEPGRAM_TTS_DEBUG") == "1"
    max_total = os.getenv("DEEPGRAM_TTS_MAX_SECONDS")
    max_total_time = float(max_total) if max_total else None
    ok = stream_tts(TTS_TEXT, debug=debug, max_total_time=max_total_time)
    if ok:
        print("TTS stream completed")
    else:
        print("TTS stream failed")


if __name__ == "__main__":
    main()
