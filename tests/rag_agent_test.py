"""
RAG Conversational Agent with Session-Based Storage

Features:
- JSON storage during conversation session
- Vector database processing only at session end
- Detailed timing logs for all operations
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
import httpx
import time
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
import logfire

# Import our complete RAG system
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.system.complete_rag import get_rag_system
from core.llm_client import (
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_with_fallbacks,
)
from core.env import load_env
from core.system_prompts import load_prompt

load_env()
logfire.configure()
logfire.instrument_pydantic_ai()

@dataclass
class RAGSession:
    """Session data for RAG agent"""
    http_client: httpx.AsyncClient
    session_started: bool = False

# Initialize models
model_settings = {
    "temperature": 0.2,
    "max_tokens": 1024,
}
openrouter_models = get_openrouter_models(settings=model_settings)
if not openrouter_models:
    raise RuntimeError("No OpenRouter API keys found. Set OPEN_ROUTER_API_KEY_1/2/3 or OPENROUTER_API_KEY in .env.")

model = openrouter_models[0]
openrouter_fallback_models = openrouter_models[1:]
groq_fallback_model = get_groq_fallback_model(settings=model_settings)

# Global variables for timing
embedding_search_time = 0.0
history_tool_time = 0.0
agent_response_time = 0.0

# RAG-enabled agent
rag_agent = Agent(
    model,
    deps_type=RAGSession,
    system_prompt=load_prompt("rag_agent"),
)
rag_agent_fallbacks: list[Agent] = []
for fallback_model in openrouter_fallback_models:
    rag_agent_fallbacks.append(
        Agent(
            fallback_model,
            deps_type=RAGSession,
            system_prompt=load_prompt("rag_agent"),
        )
    )
if groq_fallback_model:
    rag_agent_fallbacks.append(
        Agent(
            groq_fallback_model,
            deps_type=RAGSession,
            system_prompt=load_prompt("rag_agent"),
        )
    )

@rag_agent.tool
async def history_tool(ctx: RunContext[RAGSession], query: str) -> str:
    """Retrieve cross-session conversation history using RAG system"""
    global history_tool_time, embedding_search_time
    
    tool_start = time.time()
    
    try:
        rag_system = get_rag_system()
        
        # Time the embedding search specifically
        search_start = time.time()
        result = await rag_system.query_history(query)
        embedding_search_time = time.time() - search_start
        
        history_tool_time = time.time() - tool_start
        
        if result == "cannot find in history":
            return "No relevant information found in cross-session conversation history."
        else:
            # Return raw JSON for main agent to process
            return result
            
    except Exception as e:
        history_tool_time = time.time() - tool_start
        return f"Error accessing cross-session conversation history: {str(e)}"

def print_timing_logs():
    """Print timing logs after each query"""
    global embedding_search_time, history_tool_time, agent_response_time
    
    print("\nTiming Logs:")
    if agent_response_time > 0:
        print(f"Agent Response: {agent_response_time:.3f}s")
    if history_tool_time > 0:
        print(f"History Tool: {history_tool_time:.3f}s")
    if embedding_search_time > 0:
        print(f"Embedding Search: {embedding_search_time:.3f}s")

async def conversational_agent():
    """Interactive conversational agent with session-based storage"""
    global agent_response_time
    
    print("RAG Agent Started")
    print("Type 'quit', 'exit', or 'bye' to end session")
    
    # Initialize RAG system and start session
    rag_system = get_rag_system()
    await rag_system.start_session()
    print("Session started")
    
    # Show existing stats
    stats = rag_system.get_system_stats()
    print(f"Current vectors: {stats['total_vectors']}, Sessions: {stats['total_sessions']}")
    
    async with httpx.AsyncClient() as client:
        session = RAGSession(http_client=client, session_started=True)
        message_history: list[ModelMessage] | None = None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye")
                    break
                
                if not user_input:
                    continue
                
                # Time the agent response
                response_start = time.time()
                
                result = await run_agent_with_fallbacks(
                    rag_agent,
                    rag_agent_fallbacks,
                    user_input, 
                    deps=session,
                    message_history=message_history
                )
                
                agent_response_time = time.time() - response_start
                
                print(f"Agent: {result.output}")
                
                # Update short-term memory
                message_history = result.all_messages()
                
                # Add to JSON session (no vector processing yet)
                rag_system.add_conversation(user_input, result.output)
                
                # Print timing logs
                print_timing_logs()
                
                # Reset timing variables
                embedding_search_time = 0.0
                history_tool_time = 0.0
                
            except KeyboardInterrupt:
                print("\nSession interrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Process session to vector database only at the end
    print("\nProcessing session to vector database...")
    processing_start = time.time()
    await rag_system.end_session()
    processing_time = time.time() - processing_start
    
    print(f"Vector processing completed in {processing_time:.3f}s")
    
    # Show final stats
    stats = rag_system.get_system_stats()
    print(f"Final stats - Vectors: {stats['total_vectors']}, Sessions: {stats['total_sessions']}")

if __name__ == "__main__":
    asyncio.run(conversational_agent())
