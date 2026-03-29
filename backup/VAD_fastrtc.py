"""
FastRTC Voice Assistant with Real-time VAD
Uses FastRTC's built-in Silero VAD for automatic speech detection
"""

import os
import sys
import asyncio
import time
from pathlib import Path
import tempfile
from typing import Generator, Tuple
import numpy as np


from fastrtc import Stream, ReplyOnPause, AlgoOptions

# Groq for STT
from groq import Groq

# Deepgram for TTS (disabled)
# from deepgram import DeepgramClient, SpeakOptions

# LLM
from pydantic_ai import Agent

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Environment
from core.env import load_env

load_env()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY2"))
# deepgram_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

from core.llm_client import (
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
)
from core.openrouter_tts import synthesize_speech

# Configure LLM
model_settings = {
    "temperature": 0.2,
    "max_tokens": 1024,
}

openrouter_models = get_openrouter_models(settings=model_settings)
if not openrouter_models:
    raise RuntimeError("No Groq API keys found. Set OPEN_ROUTER_API_KEY_1/2/3 or OPENROUTER_API_KEY in .env.")

primary_model = openrouter_models[0]
openrouter_fallback_models = openrouter_models[1:]
groq_fallback_model = get_groq_fallback_model(settings=model_settings)
agent = Agent(
    primary_model,
    model_settings=model_settings,
    system_prompt="You are a helpful voice assistant. Keep responses concise and clear."
)
agent_fallbacks: list[Agent] = []
for fallback_model in openrouter_fallback_models:
    agent_fallbacks.append(
        Agent(
            fallback_model,
            model_settings=model_settings,
            system_prompt="You are a helpful voice assistant. Keep responses concise and clear.",
        )
    )
if groq_fallback_model:
    agent_fallbacks.append(
        Agent(
            groq_fallback_model,
            model_settings=model_settings,
            system_prompt="You are a helpful voice assistant. Keep responses concise and clear.",
        )
    )

class FastRTCVoiceAssistant:
    def __init__(self):
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        
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
        """Convert text to speech using Groq"""
        try:
            speech_path = self.temp_dir / filename
            return synthesize_speech(text, speech_path)
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
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
                import asyncio
                response = asyncio.run(self.get_llm_response(transcription))
                print(f"Assistant: {response}")
                
                # Generate TTS
                speech_file = self.text_to_speech(response)
                
                if speech_file and Path(speech_file).exists():
                    # Load and yield audio back to FastRTC
                    import scipy.io.wavfile as wavfile
                    tts_sample_rate, tts_audio_data = wavfile.read(speech_file)
                    
                    # Convert stereo to mono if needed
                    if len(tts_audio_data.shape) > 1:
                        tts_audio_data = tts_audio_data[:, 0]
                    
                    print(f"Yielding TTS audio: {len(tts_audio_data)} samples")
                    yield (tts_sample_rate, tts_audio_data.astype(np.int16))
                    
                    # Cleanup
                    Path(speech_file).unlink(missing_ok=True)
                    
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
                speech_file = self.text_to_speech(response)
                
                if speech_file:
                    # Play audio
                    try:
                        from pydub import AudioSegment
                        from pydub.playback import play
                        
                        audio = AudioSegment.from_wav(str(speech_file))
                        play(audio)
                        print("Audio played")
                    except Exception as e:
                        print(f"Could not play audio: {e}")
            else:
                print("No speech detected")
            
            # Cleanup
            if audio_path.exists():
                audio_path.unlink()
            if speech_file and Path(speech_file).exists():
                Path(speech_file).unlink()
        
        print("Goodbye!")
        self.cleanup()
    
    async def run_console_mode_async(self):
        """Async wrapper for console mode"""
        await self.run_console_mode()
    
    def run_web_interface(self):
        """Launch FastRTC web interface"""
        print("FastRTC Voice Assistant - Web Interface")
        print("Uses automatic VAD - no manual recording needed")
        
        stream = self.create_fastrtc_stream()
        
        try:
            print("Launching FastRTC web interface...")
            print("Access the voice interface in your browser")
            print("Just speak naturally - FastRTC detects pauses automatically")
            print("Press Ctrl+C to stop")
            
            # Launch web UI
            stream.ui.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                inbrowser=True
            )
            
        except KeyboardInterrupt:
            print("FastRTC stopped")
        except Exception as e:
            print(f"FastRTC error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_phone_interface(self):
        """Launch FastRTC phone interface"""
        print("FastRTC Voice Assistant - Phone Interface")
        
        stream = self.create_fastrtc_stream()
        
        try:
            print("Launching FastRTC phone interface...")
            print("A temporary phone number will be provided")
            print("Press Ctrl+C to stop")
            
            # Launch phone interface
            stream.fastphone()
            
        except KeyboardInterrupt:
            print("FastRTC phone stopped")
        except Exception as e:
            print(f"FastRTC phone error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        for file in self.temp_dir.glob("*.wav"):
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
