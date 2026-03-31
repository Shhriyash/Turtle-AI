import sys
from pathlib import Path
from pydantic_ai import Agent, WebSearchTool

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.llm_client import (
    get_openrouter_models,
    get_groq_fallback_model,
    run_agent_sync_with_fallbacks,
)
from core.env import load_env

load_env(override=True)

model_settings = {
    "temperature": 0.2,
    "max_tokens": 1024,
}

openrouter_models = get_openrouter_models(settings=model_settings)
if not openrouter_models:
    raise RuntimeError("No OpenRouter API keys found. Set OPEN_ROUTER_API_KEY_1/2/3 in .env.")

model = openrouter_models[0]
openrouter_fallback_models = openrouter_models[1:]
groq_fallback_model = get_groq_fallback_model(settings=model_settings)
agent = Agent(
    model,
    builtin_tools=[WebSearchTool()],
    system_prompt="""You are a helpful assistant with web search capabilities.

Use web search for current information, news, or recent events.
Answer general knowledge questions directly without search.""",
)
agent_fallbacks: list[Agent] = []
for fallback_model in openrouter_fallback_models:
    agent_fallbacks.append(
        Agent(
            fallback_model,
            builtin_tools=[WebSearchTool()],
            system_prompt="""You are a helpful assistant with web search capabilities.

Use web search for current information, news, or recent events.
Answer general knowledge questions directly without search.""",
        )
    )
if groq_fallback_model:
    agent_fallbacks.append(
        Agent(
            groq_fallback_model,
            builtin_tools=[WebSearchTool()],
            system_prompt="""You are a helpful assistant with web search capabilities.

Use web search for current information, news, or recent events.
Answer general knowledge questions directly without search.""",
        )
    )

while True:
    try:
        prompt = input("User: ")
        if prompt.lower() in ['exit', 'quit', 'bye']:
            break
        
        result = run_agent_sync_with_fallbacks(agent, agent_fallbacks, prompt)
        print(result.output)
        
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        print()
