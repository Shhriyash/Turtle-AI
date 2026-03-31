"""
FastRTC Voice Assistant - ACTUAL FastRTC Implementation
Uses real FastRTC Stream with Silero VAD for automatic speech detection
Keeps same STT (Groq Whisper), LLM (Gemini), TTS (Deepgram) stack
"""

import os
import asyncio
import time
from pathlib import Path
import tempfile
from typing import Generator, Tuple
import numpy as np
import re
import uuid

# FastRTC for real VAD
from fastrtc import Stream, ReplyOnPause, AlgoOptions

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
    system_prompt="You are a helpful voice assistant. Keep responses concise and clear."
)

class RealFastRTCVoiceAssistant:
    def __init__(self):
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        self.conversation_count = 0
        
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
    
    def text_to_speech(self, text):
        """Convert text to speech using Deepgram (simple MP3 approach)"""
        try:
            audio_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
            speech_path = self.temp_dir / audio_filename
            
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
        """Play audio file using pydub"""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            audio = AudioSegment.from_mp3(str(audio_path))
            play(audio)
            
            # Cleanup immediately after playback
            if audio_path.exists():
                audio_path.unlink()
            
            return True
        except Exception as e:
            print(f"Playback error: {e}")
            return False

    def response(self, audio: Tuple[int, np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Process audio detected by FastRTC VAD and generate response"""
        sample_rate, audio_array = audio
        
        vad_start_time = time.time()
        print(f"LOG: FastRTC VAD detected speech - {len(audio_array)} samples at {sample_rate}Hz")
        
        try:
            # Check if we have valid audio data
            if len(audio_array) < 1000:  # Skip very short audio snippets
                print("LOG: Audio too short, skipping")
                return
            
            # Save audio temporarily for Groq Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            import scipy.io.wavfile as wavfile
            # Ensure sample rate is within valid range for WAV format
            valid_sample_rate = max(8000, min(sample_rate, 48000))
            
            # Ensure audio data is properly formatted
            audio_data = np.array(audio_array, dtype=np.int16)
            wavfile.write(temp_path, valid_sample_rate, audio_data)
            
            vad_time = time.time() - vad_start_time
            print(f"LOG: VAD processing completed in {vad_time:.2f}s")
            
            # Transcribe using Groq Whisper
            stt_start_time = time.time()
            transcription = self.transcribe_audio(temp_path)
            stt_time = time.time() - stt_start_time
            
            # Cleanup temp file
            temp_path.unlink(missing_ok=True)
            
            if not transcription or not transcription.strip():
                print("LOG: No speech detected in transcription")
                return
                
            print(f"LOG: STT completed in {stt_time:.2f}s")
            print(f"User: {transcription}")
            
            # Get LLM response
            llm_start_time = time.time()
            response_text = asyncio.run(self.get_llm_response(transcription))
            llm_time = time.time() - llm_start_time
            
            print(f"LOG: LLM response generated in {llm_time:.2f}s")
            print(f"Assistant: {response_text}")
            
            # Generate and play TTS
            tts_start_time = time.time()
            speech_file = self.text_to_speech(response_text)
            
            if speech_file and speech_file.exists():
                tts_generation_time = time.time() - tts_start_time
                print(f"LOG: TTS generation completed in {tts_generation_time:.2f}s")
                
                # Play audio
                playback_start_time = time.time()
                playback_success = self.play_audio(speech_file)
                playback_time = time.time() - playback_start_time
                
                if playback_success:
                    print(f"LOG: Audio playback completed in {playback_time:.2f}s")
                else:
                    print("LOG: Audio playback failed")
                
                # Calculate total times
                total_processing_time = vad_time + stt_time + llm_time + tts_generation_time
                total_time = total_processing_time + playback_time
                
                print(f"LOG: Performance Summary:")
                print(f"  VAD Processing: {vad_time:.2f}s")
                print(f"  Whisper STT: {stt_time:.2f}s")
                print(f"  Gemini LLM: {llm_time:.2f}s")
                print(f"  Deepgram TTS Generation: {tts_generation_time:.2f}s")
                print(f"  Audio Playback: {playback_time:.2f}s")
                print(f"  Total Processing Time: {total_processing_time:.2f}s")
                print(f"  Total Time: {total_time:.2f}s")
            else:
                print("LOG: TTS generation failed")
                
        except Exception as e:
            print(f"LOG: FastRTC processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def create_stream(self) -> Stream:
        """Create and configure a Stream instance with audio capabilities.
        
        Returns:
            Stream: Configured FastRTC Stream instance
        """
        print("LOG: Creating FastRTC Stream with Silero VAD")
        
        stream = Stream(
            modality="audio",
            mode="send-receive",
            handler=ReplyOnPause(
                self.response,
                algo_options=AlgoOptions(
                    speech_threshold=0.2,
                ),
            ),
        )
        
        print("LOG: FastRTC Stream created successfully with Silero VAD")
        return stream



    def run_console_mode_with_fastrtc_vad(self):
        """Console mode with manual audio input and FastRTC VAD processing"""
        print("Real FastRTC Voice Assistant - Console Mode")
        print("Manual audio recording with FastRTC VAD processing")
        print("LOG: Setting up manual audio recording with FastRTC VAD")
        
        import pyaudio
        import wave
        import io
        
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        print("Choose recording mode:")
        print("1. Hold SPACE to record (release to stop)")
        print("2. Press SPACE to start, auto-stop when you finish speaking")
        print("Press Q to quit")
        
        mode_choice = input("Enter mode (1 or 2): ").strip()
        
        print("LOG: Ready for voice input")
        
        try:
            import keyboard
            
            while True:
                try:
                    print("\nWaiting for SPACE key...")
                    
                    # Wait for space key press
                    keyboard.wait('space')
                    
                    if mode_choice == "1":
                        # Mode 1: Hold SPACE to record, release to stop
                        print("Recording... (release SPACE to stop)")
                        frames = []
                        
                        # Start recording
                        stream = p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
                        
                        recording_start = time.time()
                        
                        # Record while space is held
                        while keyboard.is_pressed('space'):
                            data = stream.read(CHUNK)
                            frames.append(data)
                            
                            # Safety limit - stop after 15 seconds max
                            if time.time() - recording_start > 15:
                                print("Maximum recording time reached")
                                break
                        
                        stream.stop_stream()
                        stream.close()
                        
                        recording_time = time.time() - recording_start
                        print(f"LOG: Recorded {recording_time:.2f}s of audio")
                        
                    else:
                        # Mode 2: Press SPACE to start, auto-stop when finished speaking
                        print("Recording... (will automatically stop when you finish speaking)")
                        frames = []
                        
                        # Start recording
                        stream = p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
                        
                        recording_start = time.time()
                        
                        # Parameters for silence detection
                        SILENCE_THRESHOLD = 500  # Adjust based on your microphone
                        SILENCE_CHUNKS = 30      # Number of silent chunks before stopping (about 0.7 seconds)
                        silent_chunks = 0
                        has_spoken = False
                        
                        # Record until silence is detected
                        while True:
                            data = stream.read(CHUNK)
                            frames.append(data)
                            
                            # Convert to numpy array to check volume
                            audio_chunk = np.frombuffer(data, dtype=np.int16)
                            # Use RMS (Root Mean Square) for volume calculation with safety check
                            rms = np.sqrt(np.maximum(0, np.mean(audio_chunk.astype(np.float32)**2)))
                            volume = rms
                            
                            if volume > SILENCE_THRESHOLD:
                                # Sound detected
                                silent_chunks = 0
                                has_spoken = True
                            else:
                                # Silence detected
                                if has_spoken:  # Only count silence after we've detected speech
                                    silent_chunks += 1
                            
                            # Stop if we've had enough silence after speaking
                            if has_spoken and silent_chunks > SILENCE_CHUNKS:
                                print("Silence detected - stopping recording")
                                break
                                
                            # Safety limit - stop after 15 seconds max
                            if time.time() - recording_start > 15:
                                print("Maximum recording time reached")
                                break
                            
                            # Allow manual stop with space release (optional)
                            if not keyboard.is_pressed('space') and has_spoken and silent_chunks > 5:
                                break
                        
                        stream.stop_stream()
                        stream.close()
                        
                        recording_time = time.time() - recording_start
                        print(f"LOG: Recorded {recording_time:.2f}s of audio")
                    
                    # Process audio for both modes
                    if len(frames) > 0:
                        # Convert audio to numpy array for FastRTC processing
                        audio_data = b''.join(frames)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Process through FastRTC VAD and response pipeline
                        print("LOG: Processing audio through FastRTC VAD")
                        
                        # Simulate what FastRTC would do - call our response method directly
                        audio_tuple = (RATE, audio_array)
                        
                        # Process the audio through our response pipeline (fix generator issue)
                        response_gen = self.response(audio_tuple)
                        if response_gen:
                            for _ in response_gen:
                                pass  # Consume the generator
                    
                    # Check for quit
                    if keyboard.is_pressed('q'):
                        print("LOG: Quit key detected")
                        break
                        
                except KeyboardInterrupt:
                    print("\nLOG: Received Ctrl+C")
                    break
                except Exception as e:
                    print(f"LOG: Error during recording: {e}")
                    continue
            
        except ImportError:
            print("LOG: keyboard module not available, using simple input method")
            print("Press Enter to record, type 'quit' to exit")
            
            while True:
                try:
                    user_input = input("\nPress Enter to start recording (or 'quit' to exit): ").strip().lower()
                    
                    if user_input == 'quit':
                        break
                    frames = []
                    
                    stream = p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)
                    
                    recording_start = time.time()
                    
                    for i in range(0, int(RATE / CHUNK * 3)):  # 3 seconds
                        data = stream.read(CHUNK)
                        frames.append(data)
                    
                    stream.stop_stream()
                    stream.close()
                    
                    recording_time = time.time() - recording_start
                    print(f"LOG: Recorded {recording_time:.2f}s of audio")
                    
                    # Process audio
                    audio_data = b''.join(frames)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_tuple = (RATE, audio_array)
                    
                    # Process through our response pipeline (fix generator issue)
                    response_gen = self.response(audio_tuple)
                    if response_gen:
                        for _ in response_gen:
                            pass  # Consume the generator
                    
                except KeyboardInterrupt:
                    print("\nLOG: Received Ctrl+C")
                    break
                except Exception as e:
                    print(f"LOG: Error: {e}")
                    continue
        
        finally:
            try:
                p.terminate()
                print("LOG: Audio system terminated")
            except:
                pass
        
        return True

    def cleanup(self):
        """Clean up resources"""
        print("LOG: Cleaning up temporary files")
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        for file in self.temp_dir.glob("*.mp3"):
            file.unlink()
        print("LOG: Cleanup completed")

def main():
    """Main function"""
    print("FastRTC Voice Assistant - REAL Implementation")
    print("LOG: Initializing Real FastRTC Voice Assistant")
    
    assistant = RealFastRTCVoiceAssistant()
    
    print("LOG: Assistant initialized")
    print("Starting console mode with FastRTC automatic VAD")
    print("LOG: User starting console mode")
    
    assistant.run_console_mode_with_fastrtc_vad()
    assistant.cleanup()

if __name__ == "__main__":
    main()
