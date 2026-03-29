"""
Clean Voice Assistant with FastRTC VAD
Minimal implementation with only essential features
"""

import os
import sys
import asyncio
import time
from pathlib import Path
import tempfile
from typing import Generator, Tuple
# Audio recording
import pyaudio
import wave
import numpy as np
# Groq for STT
from groq import Groq
# Deepgram for TTS (primary via core/openrouter_tts.py)
# from deepgram import DeepgramClient, SpeakOptions
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
AGENT_PROMPT = load_prompt("vad_simple_agent")

openrouter_models = get_openrouter_models(settings=model_settings)
if not openrouter_models:
    raise RuntimeError("No OpenRouter API keys found. Set OPEN_ROUTER_API_KEY_1/2/3 or OPENROUTER_API_KEY in .env.")

primary_model = openrouter_models[0]
openrouter_fallback_models = openrouter_models[1:]
groq_fallback_model = get_groq_fallback_model(settings=model_settings)
agent = Agent(
    primary_model,
    model_settings=model_settings,
    system_prompt=AGENT_PROMPT,   
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

# Audio config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 8
SILENCE_DURATION = 0.8

class VoiceAssistant:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.temp_dir = TEMP_AUDIO_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def record_audio(self, filename="input.wav"):
        """Record audio with VAD"""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        silent_chunks = 0
        audio_started = False
        speech_energy_threshold = 50
        chunk_count = 0
        
        try:
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
                chunk_count += 1
                
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms_energy = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                
                if rms_energy > speech_energy_threshold:
                    silent_chunks = 0
                    audio_started = True
                elif audio_started:
                    silent_chunks += 1
                
                if audio_started and silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
                    break
                    
                if not audio_started and chunk_count > (1.5 * RATE / CHUNK):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
        
        audio_path = self.temp_dir / filename
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        return str(audio_path)
    
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
    
    def text_to_speech(self, text, filename="output.wav"):
        """Convert text to speech using Groq TTS"""
        try:
            speech_path = self.temp_dir / filename
            return synthesize_speech(text, speech_path)
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    def play_audio(self, audio_path):
        """Play audio file"""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            audio = AudioSegment.from_file(str(audio_path))
            
            play(audio)
            return True
        except Exception as e:
            print(f"Playback error: {e}")
            return False
    

    
    async def run_console_mode(self):
        """Run in console mode with manual recording"""
        print("Commands: 'quit' to exit")
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("\nPress Enter to speak: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                conversation_count += 1
                total_start = time.time()
                
                # Record audio
                record_start = time.time()
                audio_file = self.record_audio(f"input_{conversation_count}.wav")
                record_time = time.time() - record_start
                
                # Transcribe
                stt_start = time.time()
                user_text = self.transcribe_audio(audio_file)
                stt_time = time.time() - stt_start
                
                if not user_text:
                    print("Could not understand speech")
                    continue
                
                print(f"You: {user_text}")
                
                # Get response
                llm_start = time.time()
                response_text = await self.get_llm_response(user_text)
                llm_time = time.time() - llm_start
                print(f"Assistant: {response_text}")
                
                # Generate and play TTS
                tts_start = time.time()
                speech_file = self.text_to_speech(response_text, f"output_{conversation_count}.wav")
                tts_time = time.time() - tts_start
                
                if speech_file:
                    self.play_audio(speech_file)
                
                # Display timing stats
                total_time = time.time() - total_start
                print(f"Times: Record={record_time:.2f}s | STT={stt_time:.2f}s | LLM={llm_time:.2f}s | TTS={tts_time:.2f}s | Total={total_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        self.cleanup()
    

    
    def cleanup(self):
        """Clean up resources"""
        self.audio.terminate()
        
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        for file in self.temp_dir.glob("*.mp3"):
            file.unlink()

async def main():
    """Main function"""
    print("Voice Assistant")
    
    assistant = VoiceAssistant()
    
    # Run console mode only
    await assistant.run_console_mode()

if __name__ == "__main__":
    # Check dependencies
    try:
        import pyaudio
    except ImportError:
        print("PyAudio required: pip install pyaudio")
        exit(1)
    
    asyncio.run(main())
