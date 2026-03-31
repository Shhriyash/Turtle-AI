"""
Clean Voice Assistant with FastRTC VAD
Minimal implementation with only essential features
"""

import os
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
# Deepgram for TTS  
from deepgram import DeepgramClient, SpeakOptions
# Gemini for LLM
from google.genai.types import HarmBlockThreshold, HarmCategory
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModelSettings
from vertex_genai_client import get_vertex_model

# Environment
from dotenv import load_dotenv
load_dotenv()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY2"))
deepgram_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

# Configure Gemini
settings = GoogleModelSettings(
    temperature=0.2,
    max_tokens=1024,
    google_thinking_config={'thinking_budget': 2048},
    google_safety_settings=[{
        'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }]
)

gemini_model = get_vertex_model('gemini-2.5-flash')
agent = Agent(
    gemini_model,
    model_settings=settings,
    system_prompt="""You are a helpful voice assistant. Keep responses concise and clear.
                     Do not use asterisks (*) or other markdown formatting in your responses.
                     Do no use emojis.""",   
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
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        
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
        """Get response from Gemini LLM"""
        try:
            result = await agent.run(text)
            return result.output
        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, I couldn't process that."
    
    def text_to_speech(self, text, filename="output.mp3"):
        """Convert text to speech using Deepgram"""
        try:
            speech_path = self.temp_dir / filename
            
            options = SpeakOptions(
                model="aura-2-draco-en",
                encoding="mp3"
            )
            
            text_payload = {"text": text}
            
            response = deepgram_client.speak.rest.v("1").save(
                str(speech_path),
                text_payload,
                options
            )
            
            return speech_path
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    def play_audio(self, audio_path):
        """Play audio file"""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            if str(audio_path).endswith('.mp3'):
                audio = AudioSegment.from_mp3(str(audio_path))
            else:
                audio = AudioSegment.from_wav(str(audio_path))
            
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
                speech_file = self.text_to_speech(response_text, f"output_{conversation_count}.mp3")
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
