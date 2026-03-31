from pydantic_ai import Agent, WebSearchTool
from dotenv import load_dotenv

from vertex_genai_client import get_vertex_model

load_dotenv(override=True)

# model = GroqModel('compound-beta')
model = get_vertex_model('gemini-2.5-flash')
agent = Agent(
    model,
    builtin_tools=[WebSearchTool()],
    system_prompt="""You are a helpful assistant with web search capabilities.

Use web search for current information, news, or recent events.
Answer general knowledge questions directly without search.""",
)

while True:
    try:
        prompt = input("User: ")
        if prompt.lower() in ['exit', 'quit', 'bye']:
            break
        
        result = agent.run_sync(prompt)
        print(result.output)
        
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        print()
