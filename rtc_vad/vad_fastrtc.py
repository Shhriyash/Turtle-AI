"""
FastRTC Voice Assistant with Real-time VAD.
Uses FastRTC's built-in Silero VAD for automatic speech detection.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
import tempfile
from typing import Generator, Tuple
import numpy as np
import re


from fastrtc import Stream, ReplyOnPause, AlgoOptions

# Groq for STT
from groq import Groq

# Deepgram for TTS (primary via core/openrouter_tts.py)
# from deepgram import DeepgramClient, SpeakOptions, SpeakWebSocketEvents
# import sounddevice as sd

# LLM
from pydantic_ai import Agent
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.llm_client import (
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
)
from core.paths import TEMP_AUDIO_DIR, ensure_dirs
from core.openrouter_tts import synthesize_speech
from core.env import load_env
from core.system_prompts import load_prompt

# Environment
load_env()
ensure_dirs()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2"))
# deepgram_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

# Configure LLM
model_settings = {
    "temperature": 0.2,
    "max_tokens": 1024,
}
AGENT_PROMPT = load_prompt("vad_fastrtc_agent")

openrouter_models = get_openrouter_models(settings=model_settings)
if not openrouter_models:
    raise RuntimeError("No OpenRouter API keys found. Set OPEN_ROUTER_API_KEY_1/2/3 or OPENROUTER_API_KEY in .env.")

primary_model = openrouter_models[0]
openrouter_fallback_models = openrouter_models[1:]
groq_fallback_model = get_groq_fallback_model(settings=model_settings)
agent = Agent(
    primary_model,
    model_settings=model_settings,
    system_prompt=AGENT_PROMPT
)
agent_fallbacks: list[Agent] = []
for fallback_model in openrouter_fallback_models:
    agent_fallbacks.append(
        Agent(
            fallback_model,
            model_settings=model_settings,
            system_prompt=AGENT_PROMPT,
        )
    )
if groq_fallback_model:
    agent_fallbacks.append(
        Agent(
            groq_fallback_model,
            model_settings=model_settings,
            system_prompt=AGENT_PROMPT,
        )
    )

def run_coro_sync(coro):
    """Run a coroutine from sync context, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

class FastRTCVoiceAssistant:
    def __init__(self):
        self.temp_dir = TEMP_AUDIO_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def chunk_by_sentence(self, text):
        """Split text at sentence boundaries for optimized streaming TTS"""
        # Split text at sentence boundaries (periods, question marks, exclamation points)
        # while preserving the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove any empty chunks and strip whitespace
        chunks = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        # If no sentence boundaries found, split by length (fallback)
        if len(chunks) <= 1 and len(text) > 100:
            # Split long text into ~80 character chunks at word boundaries
            words = text.split()
            chunks = []
            current_chunk = ""
            
            for word in words:
                if len(current_chunk + " " + word) > 80 and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk += (" " if current_chunk else "") + word
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
        
    def transcribe_audio(self, audio_path):
        """Convert audio to text using Groq Whisper"""
        try:
            with open(audio_path, "rb") as file:
                filename = Path(audio_path).name
                transcription = groq_client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
            return transcription.text
        except Exception as e:
            print(f"STT error: {e}")
            return None
    
    async def get_llm_response(self, text):
        """Get response from the LLM"""
        try:
            result = await run_agent_with_fallbacks(agent, agent_fallbacks, text)
            return result.output
        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, I couldn't process that."
    
    def stream_text_to_speech(self, text):
        """Generate and play TTS audio using Groq TTS"""
        try:
            print(f"Generating TTS: {text[:50]}...")
            speech_path = synthesize_speech(text, self.temp_dir / "output.wav")

            try:
                from pydub import AudioSegment
                from pydub.playback import play

                audio = AudioSegment.from_file(str(speech_path))
                # Add short silence padding to avoid clipped start/end
                pad = AudioSegment.silent(duration=120)
                audio = pad + audio + pad
                audio = audio.fade_in(20).fade_out(40)
                play(audio)
            except Exception:
                if speech_path.suffix.lower() == ".wav":
                    import sounddevice as sd
                    import scipy.io.wavfile as wavfile

                    rate, data = wavfile.read(str(speech_path))
                    pad_len = int(rate * 0.12)
                    if data.ndim == 1:
                        pad = np.zeros(pad_len, dtype=data.dtype)
                        data = np.concatenate([pad, data, pad])
                    else:
                        pad = np.zeros((pad_len, data.shape[1]), dtype=data.dtype)
                        data = np.concatenate([pad, data, pad])
                    sd.play(data, rate)
                    sd.wait()
                else:
                    raise

            if speech_path.exists():
                speech_path.unlink()
            return True
        except Exception as e:
            print(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def text_to_speech(self, text, filename="output.wav"):
        """TTS method for minimal latency"""
        success = self.stream_text_to_speech(text)
        return "streamed" if success else None
    
    def create_fastrtc_stream(self):
        """Create FastRTC stream with automatic VAD"""
        def response_generator(audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
            """Process audio and generate response using FastRTC's automatic VAD"""
            sample_rate, audio_array = audio
            print(f"FastRTC VAD detected speech: {len(audio_array)} samples at {sample_rate}Hz")
            
            try:
                # Check if we have valid audio data
                if len(audio_array) < 100:  # Skip very short audio snippets
                    print("Audio too short, skipping...")
                    return
                
                # Save audio temporarily for Groq Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                import scipy.io.wavfile as wavfile
                # Ensure sample rate is within valid range for WAV format (16-bit WAV limit)
                valid_sample_rate = max(8000, min(sample_rate, 48000))  # Between 8kHz-48kHz
                
                # Ensure audio data is properly formatted
                audio_data = np.array(audio_array, dtype=np.int16)
                wavfile.write(temp_path, valid_sample_rate, audio_data)
                
                # Transcribe using Groq Whisper
                transcription = self.transcribe_audio(temp_path)
                temp_path.unlink(missing_ok=True)
                
                if not transcription or not transcription.strip():
                    print("No speech detected")
                    return
                    
                print(f"User: {transcription}")
                
                # Get LLM response
                response = run_coro_sync(self.get_llm_response(transcription))
                print(f"Assistant: {response}")
                
                # Generate TTS using streaming (audio plays automatically)
                success = self.stream_text_to_speech(response)
                
                if success:
                    print("TTS streaming completed")
                    # Note: For FastRTC integration, we could collect audio chunks
                    # and yield them back, but for now streaming plays directly
                else:
                    print("TTS streaming failed")
                    
            except Exception as e:
                print(f"FastRTC processing error: {e}")
                import traceback
                traceback.print_exc()
        
        # Create FastRTC stream with automatic VAD
        stream = Stream(
            modality="audio",
            mode="send-receive",
            handler=ReplyOnPause(
                response_generator,
                algo_options=AlgoOptions(
                    speech_threshold=0.2,  # Sensitive speech detection
                )
            )
        )
        
        return stream
    
    async def run_console_mode(self):
        """Console mode with FastRTC processing"""
        print("FastRTC Voice Assistant - Console Mode")
        print("Using FastRTC for automatic speech detection")
        print("Press Enter to start listening, or 'quit' to exit")
        
        conversation_count = 0
        
        while True:
            user_input = input("\nPress Enter to speak (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            conversation_count += 1
            print(f"\n--- Conversation {conversation_count} ---")
            
            # Simulate FastRTC processing with manual recording for now
            print("Recording... (speak now)")
            
            # Use simple recording since full FastRTC requires WebRTC
            import pyaudio
            import wave
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16 
            CHANNELS = 1
            RATE = 44100
            RECORD_SECONDS = 5
            
            audio = pyaudio.PyAudio()
            
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS, 
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save audio
            audio_path = self.temp_dir / f"input_{conversation_count}.wav"
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            # Process through pipeline
            print("Transcribing...")
            transcription = self.transcribe_audio(audio_path)
            
            if transcription and transcription.strip():
                print(f"User: {transcription}")
                
                print("Getting response...")
                response = await self.get_llm_response(transcription)
                print(f"Assistant: {response}")
                
                print("Generating speech...")
                success = self.stream_text_to_speech(response)
                
                if not success:
                    print("TTS streaming failed")
            else:
                print("No speech detected")
            
            # Cleanup
            if audio_path.exists():
                audio_path.unlink()
            # No TTS file cleanup needed - streaming doesn't create files
        
        print("Goodbye!")
        self.cleanup()
    
    async def run_console_mode_async(self):
        """Async wrapper for console mode"""
        await self.run_console_mode()
    

    

    
    def cleanup(self):
        """Clean up resources"""
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        for file in self.temp_dir.glob("*.mp3"):
            file.unlink()

def main():
    """Main function"""
    print("FastRTC Voice Assistant")
    print("Automatic VAD with real-time speech detection")
    
    assistant = FastRTCVoiceAssistant()
    
    # Run console mode directly
    asyncio.run(assistant.run_console_mode_async())
    
    assistant.cleanup()

if __name__ == "__main__":
    main()
