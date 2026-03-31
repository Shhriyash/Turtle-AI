"""
Turtle - Personal Assistant with Web Search and URL Context Capabilities

Enhanced assistant with real-time web search, URL analysis, and conversation memory.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional, Generator, Tuple
import httpx
import time
import tempfile
from pathlib import Path
import numpy as np
from pydantic_ai import Agent, RunContext, WebSearchTool
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits, RunUsage
from dotenv import load_dotenv
import logfire

# Basic VAD with STT

# Groq for STT
from groq import Groq

# custom URL tools package
from url_tools import fetch_url_content_async
# Email tools
from email_tools.config import create_email_tool_from_env
# RAG system for conversation memory
from complete_rag import get_rag_system
from vertex_genai_client import get_vertex_model



load_dotenv(override=True)
logfire.configure()  
logfire.instrument_pydantic_ai()

# Initialize Groq client for STT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY2"))  # Uses GROQ_API_KEY2 from .env


@dataclass
class SharedState:
    """Shared state across all agents - now using UrlState for URL operations"""
    http_client: httpx.AsyncClient


class TurtleVoiceProcessor:
    """Voice processing class for Turtle assistant with STT"""
    
    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
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
    
    async def process_voice_input(self, audio: Tuple[int, np.ndarray]) -> Optional[str]:
        """Process recorded audio and return transcription"""
        sample_rate, audio_array = audio
        
        processing_start_time = time.time()
        print(f"LOG: Processing audio - {len(audio_array)} samples at {sample_rate}Hz")
        
        try:
            # Check if we have valid audio data
            if len(audio_array) < 1000:  # Skip very short audio snippets
                print("LOG: Audio too short, skipping")
                return None
            
            # Save audio temporarily for Groq Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            import scipy.io.wavfile as wavfile
            # Ensure sample rate is within valid range for WAV format
            valid_sample_rate = max(8000, min(sample_rate, 48000))
            
            # Ensure audio data is properly formatted
            audio_data = np.array(audio_array, dtype=np.int16)
            wavfile.write(temp_path, valid_sample_rate, audio_data)
            
            processing_time = time.time() - processing_start_time
            print(f"LOG: Audio processing completed in {processing_time:.2f}s")
            
            # Transcribe using Groq Whisper
            stt_start_time = time.time()
            transcription = self.transcribe_audio(temp_path)
            stt_time = time.time() - stt_start_time
            
            # Cleanup temp file
            temp_path.unlink(missing_ok=True)
            
            if not transcription or not transcription.strip():
                print("LOG: No speech detected in transcription")
                return None
                
            print(f"LOG: STT completed in {stt_time:.2f}s")
            print(f"User: {transcription}")
            
            return transcription
                
        except Exception as e:
            print(f"LOG: Voice processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up resources"""
        print("LOG: Cleaning up temporary files")
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        print("LOG: Cleanup completed")

# Set GROQ_API_KEY to use GROQ_API_KEY2 for consistency
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY2", os.getenv("GROQ_API_KEY", ""))

# Model settings to disable thinking
model_settings = GoogleModelSettings(google_thinking_config={'thinking_budget': 0})


model = get_vertex_model('gemini-2.5-flash')  #Main assistant
# model2 = GroqModel('llama-3.1-8b-instant')   # Email agent  
# web_search_model = GroqModel('compound-beta')  # Web search agent
model2 = get_vertex_model('gemini-2.5-flash')   # Email agent  
web_search_model = get_vertex_model('gemini-2.5-flash')  # Web search agent
usage_limits = UsageLimits(request_limit=30)

# Global voice processor instance
voice_processor = None


# Web search agent with real-time information capabilities (thinking disabled)
web_search_agent = Agent(
    web_search_model,
    deps_type=SharedState,
    model_settings=model_settings,
    builtin_tools=[WebSearchTool()],
    system_prompt="""You are a helpful assistant with web search capabilities.
                    Use web search for any type of real time search , retreiving current information, news, or recent events/activities across globe.
                """,
                        )# Email specialist agent for handling email operations
email_agent = Agent(
    model2,
    deps_type=SharedState,
    system_prompt="""You are an email specialist assistant. Your job is to help Shriyash send emails quickly and efficiently.

SIMPLE WORKFLOW:
1. Extract email addresses, subject, and content from user input
2. Use send_email tool with all the information
3. If information is missing, ask for it directly (don't use tools to store partial info)

Available Tools:
- send_email(recipients, subject, content): Send email immediately when you have all info

IMPORTANT: 
- Always try to extract recipients, subject, and content from the user's message
- Recipients can be multiple email addresses separated by commas
- Structure email content professionally like switching paragraphs, line etc.
- Send email immediately when you have all required information
- Be direct and efficient - don't overcomplicate the process
- You have access to conversation history to understand context and avoid repeating questions"""
)

@email_agent.tool
async def send_email(ctx: RunContext[SharedState], recipients: str, subject: str, content: str) -> str:
    """Send an email with the provided recipients, subject, and content
    
    Args:
        recipients: Email addresses (single email or comma-separated list for multiple recipients)
        subject: Email subject line
        content: Email content/message
    
    Returns:
        Success message with details or error message
    """
    try:
        # Format content professionally
        enhanced_content = f"""

{content}

Best regards,
TurtleAI"""
        
        # Create email tool instance with environment configuration
        email_tool = create_email_tool_from_env()
        if not email_tool:
            return "Email configuration missing. Please set up TURTLE_EMAIL_NAME, TURTLE_EMAIL_ADDRESS, and TURTLE_EMAIL_PASSKEY environment variables."
        
        result = email_tool.send_email(
            receiver=recipients,
            subject=subject,
            body=enhanced_content,
            content_type="plain"
        )
        
        # Check if result indicates an error
        if result.startswith("error:"):
            return f"Failed to send email: {result}"
        
        return f"Email sent successfully!\n\nTo: {recipients}\nSubject: {subject}\n\n{result}"
        
    except Exception as e:
        return f"Failed to send email: {str(e)}"

    

# Main assistant agent with enhanced delegation
main_assistant = Agent(
    model,
    output_type=str,
    system_prompt="""Personality: 
                    1)You are Turtle, Shriyash's personal assistant.With good ettiquetes to your responses.
                    2)Add a dash of humor to your responses.
                    Behaviour: 
                    1)Never mention you are an AI model/Large language Model.
                    2)Do not use asteriks(*) or emojis in your responses.
                    3)Dont overexaggerate humor.
                    Tools: 
                    1)Use search_web tool for any type of real time search, retrieving current information, news or recent events/activities across globe.
                    2)Use search_url tool for analyzing and extracting detailed content from web pages and URLs.
                    3)Use send_email_assistant tool for sending emails to one or multiple recipients.
                    4)Use history_tool when users ask about history/memory requests,Information from previous sessions,past conversations, previous discussions, or want to remember something we talked about before.
                        The history_tool returns raw JSON data containing conversation chunks with similarity scores.
                        Your task is to:
                        1. Identify the intent of the user's query regarding past conversations
                        2. Extract the actual conversation content from the JSON chunks
                        3. Present the information naturally without mentioning technical details like scores or chunks
                """
)

@main_assistant.tool
async def search_web(ctx: RunContext, query: str) -> str:
    """Search the web for current information"""
    print("\nSEARCHING: Web search for current information")
    result = await web_search_agent.run(
        f"Search for: {query}",
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return result.output

@main_assistant.tool
async def search_url(ctx: RunContext[SharedState], url: str) -> str:
    """Analyze and extract detailed content from a URL using custom extraction tool"""
    print(f"\nANALYZING: URL content extraction from {url}")
    
    # Use our custom URL extraction tool
    result = await fetch_url_content_async(ctx.deps.http_client, url)
    
    # Return formatted string representation
    return result.to_formatted_string()

@main_assistant.tool
async def send_email_assistant(ctx: RunContext[SharedState], query: str) -> str:
    """Send emails using the email specialist agent. Pass the complete user request about sending emails."""
    print(f"\nEMAIL: Delegating to email specialist")
    
    result = await email_agent.run(
        query,
        deps=ctx.deps,
        usage=ctx.usage
    )
    
    return result.output

@main_assistant.tool
async def history_tool(ctx: RunContext[SharedState], query: str) -> str:
    """Search conversation history for past discussions and information"""
    try:
        rag_system = get_rag_system()
        result = await rag_system.query_history(query)
        
        if result == "cannot find in history":
            return "No relevant information found in our previous conversations."
        else:
            return result
            
    except Exception as e:
        return "Unable to access conversation history at the moment."


async def text_chat(state: SharedState, rag_system):
    """Text chat mode for interactive typing-based conversations"""
    print("\n" + "="*50)
    print("SWITCHED TO TEXT MODE")
    print("Type your messages and press Enter")
    print("Type 'quit', 'exit', 'bye', or 'voice mode' to exit")
    print("="*50 + "\n")
    
    try:
        message_history: list[ModelMessage] | None = None
        usage = RunUsage()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("Turtle: Goodbye! Returning to voice mode...")
                    break
                elif user_input.lower() in ['voice mode', 'switch to voice mode', 'run voice mode']:
                    print("Turtle: Switching back to voice mode...")
                    break
                
                if not user_input:
                    continue
                
                # Run with complete message history for conversation continuity
                response = await main_assistant.run(
                    user_input,
                    deps=state,
                    message_history=message_history,
                    usage=usage,
                    usage_limits=usage_limits
                )
                
                print(f"Turtle: {response.output}")
                
                # Update message history with complete conversation for continuity
                message_history = response.all_messages()
                
                # Add conversation to RAG system for long-term memory
                rag_system.add_conversation(user_input, response.output)
                
                # Show usage information periodically
                if usage.requests % 5 == 0 and usage.requests > 0:
                    print(f"\n[Usage: {usage.requests} requests, {usage.total_tokens} tokens]")
                
            except KeyboardInterrupt:
                print("\nTurtle: Returning to voice mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Let's try again...")
                # Message history is preserved even after errors
                
    except Exception as e:
        print(f"Text mode error: {e}")


async def voice_response_handler(audio: Tuple[int, np.ndarray], state: SharedState, rag_system) -> bool:
    """Handle voice input and generate response using main assistant
    
    Returns:
        bool: True if mode switch to text is requested, False otherwise
    """
    global voice_processor
    
    # Process voice input to get transcription
    transcription = await voice_processor.process_voice_input(audio)
    
    if not transcription:
        return False
    
    # Check for mode switch commands
    switch_phrases = [
        'switch to text mode', 'run text mode', 'text mode', 
        'switch to text', 'run text', 'text chat',
        'use text mode', 'start text mode'
    ]
    
    transcription_lower = transcription.lower().strip()
    for phrase in switch_phrases:
        if phrase in transcription_lower:
            print(f"Turtle: Sure! Switching to text mode now...")
            return True
    
    try:
        # Get response from main assistant
        llm_start_time = time.time()
        response = await main_assistant.run(
            transcription,
            deps=state,
            usage=RunUsage()
        )
        llm_time = time.time() - llm_start_time
        
        print(f"LOG: LLM response generated in {llm_time:.2f}s")
        print(f"Turtle: {response.output}")
        
        # Add conversation to RAG system for memory
        rag_system.add_conversation(transcription, response.output)
        
        return False
        
    except Exception as e:
        print(f"LOG: LLM processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def voice_chat():
    """Voice chat mode using FastRTC VAD and STT with main assistant"""
    global voice_processor
    
    async with httpx.AsyncClient() as client:
        state = SharedState(http_client=client)
        voice_processor = TurtleVoiceProcessor(state)
        
        # Initialize RAG system
        rag_system = get_rag_system()
        await rag_system.start_session()
        print("Manual audio recording with VAD processing")
        print("LOG: Setting up manual audio recording with VAD")
        
        import pyaudio
        import keyboard
        
        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        print("Hold SPACE to record, release to stop (Press Ctrl+C to quit)")
        print("LOG: Ready for voice input")
        
        try:
            while True:
                try:
                    print("\nWaiting for SPACE key...")
                    
                    # Wait for space key press
                    keyboard.wait('space')
                    
                    # Hold SPACE to record, release to stop
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
                    
                    # Process audio
                    if len(frames) > 0:
                        # Convert audio to numpy array for processing
                        audio_data = b''.join(frames)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Process through voice handler
                        print("LOG: Processing audio through VAD")
                        
                        audio_tuple = (RATE, audio_array)
                        
                        # Process the audio through our voice handler
                        should_switch_to_text = await voice_response_handler(audio_tuple, state, rag_system)
                        
                        # If mode switch is requested, enter text mode
                        if should_switch_to_text:
                            # Cleanup current voice session
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            print("LOG: Audio system paused for text mode")
                            
                            # Enter text mode
                            await text_chat(state, rag_system)
                            
                            # Restart voice mode after text chat ends
                            print("\nReturning to voice mode...")
                            print("Hold SPACE to record, release to stop (Press Ctrl+C to quit)")
                            print("LOG: Ready for voice input")
                            
                            # Reinitialize PyAudio
                            p = pyaudio.PyAudio()
                            continue
                    
                except KeyboardInterrupt:
                    print("\nLOG: Received Ctrl+C")
                    break
                except Exception as e:
                    print(f"LOG: Error during recording: {e}")
                    continue
        
        finally:
            try:
                p.terminate()
                print("LOG: Audio system terminated")
            except:
                pass
            
            # Cleanup
            if voice_processor:
                voice_processor.cleanup()
            
            # End RAG session
            await rag_system.end_session()



async def main():
    print("Welcome to Turtle Assistant!")
    print("Say 'switch to text mode' or 'run text mode' to switch to typing mode")
    
    try:
        await voice_chat()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        # RAG session cleanup is handled in voice_chat function
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())
