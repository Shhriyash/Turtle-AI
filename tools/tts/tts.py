import time
import queue
import threading
import numpy as np

# NOTE: `sounddevice` (PortAudio) is imported lazily inside stream_tts(), not at
# module top level. The web server (apps.turtle_server) imports this package via
# tools.tts.__init__ -> tools.tts.client, and it synthesizes speech over REST
# (returns bytes to the browser) — it never plays audio server-side. Importing
# sounddevice at module load would crash boot on headless containers (e.g. the
# Render/Docker image) that have no PortAudio library. Playback is only used by
# the local CLI voice path, which imports it on demand below.

from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text

from tools.tts.client import get_deepgram_client

TTS_TEXT = "Hello, this is a streaming text to speech example using Deepgram."
MODEL = "aura-2-orion-en"
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
) -> bool:
    import sounddevice as sd  # lazy: PortAudio only needed for local playback

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    done_event = threading.Event()
    started_audio = threading.Event()
    last_audio_time = {"t": 0.0}

    def on_message(message) -> None:
        if isinstance(message, bytes):
            if not message:
                return
            array = np.frombuffer(message, dtype=np.int16)
            audio_queue.put(array)
            last_audio_time["t"] = time.time()
            started_audio.set()
        else:
            msg_type = getattr(message, "type", "Unknown")
            print(f"Received {msg_type} event")

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

    def idle_monitor():
        while not done_event.is_set():
            if not started_audio.is_set():
                if time.time() - last_audio_time["t"] >= start_timeout and last_audio_time["t"] > 0:
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
                dg_connection.send_text(SpeakV1Text(text=text))
                dg_connection.send_flush()
                last_audio_time["t"] = time.time()

                monitor = threading.Thread(target=idle_monitor, daemon=True)
                monitor.start()

                while not done_event.is_set():
                    time.sleep(0.05)

                dg_connection.send_close()

        return started_audio.is_set()
    except Exception as e:
        print(f"Streaming TTS error: {e}")
        return False


def main():
    ok = stream_tts(TTS_TEXT)
    if ok:
        print("TTS stream completed")
    else:
        print("TTS stream failed")


if __name__ == "__main__":
    main()
