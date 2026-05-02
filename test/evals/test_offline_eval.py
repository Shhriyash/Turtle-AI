"""
test/evals/test_offline_eval.py
-------------------------------
H2: Offline agent evaluation using pytest.
Runs 10 standard prompts directly through the agent graph without websockets.
"""

import pytest
import json
import httpx
from typing import Any

from core.llm_client import run_agent_with_fallbacks
from pydantic_ai.messages import ModelRequest, ToolCallPart
from apps.turtle_server import SharedState, agents_mgr
from core.paths import TASK_HISTORY_FILE, MEMORY_GRAPH_FILE, MEMORY_PROFILE_FILE, MEMORY_EVENTS_FILE, MEMORY_EPISODES_FILE, MEMORY_STATE_FILE, personal_memory_dir
from core.session_store import SessionStore
from core.graph_store import GraphStore
from core.memory_store import MemoryStore
from core.personal_memory_prompt import PersonalMemoryPromptBuilder, PersonalMemoryPromptConfig
from core.confirmation_gate import ConfirmationGate
from rag.system.complete_rag import TurtleRAGSystem
from core.storage.local.faiss_store import FAISSVectorStore

OFFLINE_PROMPTS = [
    {"id": "math_1", "category": "math", "prompt": "What is 25 * 4?", "expected_tool_calls": []},
    {"id": "search_1", "category": "search", "prompt": "Who is the president of the United States?", "expected_tool_calls": ["search_web"]},
    {"id": "search_2", "category": "search", "prompt": "What is the weather in New York?", "expected_tool_calls": ["search_web"]},
    {"id": "extract_1", "category": "extraction", "prompt": "Summarize https://example.com", "expected_tool_calls": ["search_url"]},
    {"id": "math_2", "category": "math", "prompt": "Calculate 10% of 500.", "expected_tool_calls": []},
    {"id": "search_3", "category": "search", "prompt": "Latest news on artificial intelligence", "expected_tool_calls": ["search_web"]},
    {"id": "extract_2", "category": "extraction", "prompt": "What are the headings from https://example.com", "expected_tool_calls": ["search_url"]},
    {"id": "search_4", "category": "search", "prompt": "What time is it in London right now?", "expected_tool_calls": ["search_web"]},
    {"id": "math_3", "category": "math", "prompt": "Square root of 144", "expected_tool_calls": []},
    {"id": "extract_3", "category": "extraction", "prompt": "What does https://example.com say?", "expected_tool_calls": ["search_url"]},
]

class MockWebSocket:
    def __init__(self):
        self.messages = []
        
    async def send_text(self, text: str):
        self.messages.append(json.loads(text))

@pytest.fixture
def offline_state():
    user_id = "test_offline"
    from core.personal_memory_store import PersonalMemoryStore
    from core.memory_journal import JournalStore
    from core.task_history import TaskHistoryStore
    from core.retrieval_broker import RetrievalBroker
    from core.storage.local.faiss_store import FAISSVectorStore
    
    personal_memory_store = PersonalMemoryStore(user_id=user_id)
    journal_store = JournalStore(user_id=user_id)
    task_history_store = TaskHistoryStore(history_path=TASK_HISTORY_FILE)
    vector_store = FAISSVectorStore()
    
    rag_system = TurtleRAGSystem()
    retrieval_broker = RetrievalBroker(
        store=personal_memory_store,
        task_store=task_history_store,
        rag_system=rag_system,
        vector_store=vector_store,
    )
    
    session_store = SessionStore()
    graph_store = GraphStore(graph_path=MEMORY_GRAPH_FILE)
    memory_store = MemoryStore(
        profile_path=MEMORY_PROFILE_FILE,
        events_path=MEMORY_EVENTS_FILE,
        episodes_path=MEMORY_EPISODES_FILE,
        state_path=MEMORY_STATE_FILE,
        graph_store=graph_store,
        flush_turns=1,
        flush_tokens=1000,
        profile_max_lines=100,
        write_enabled=False,
    )
    personal_memory_prompt = PersonalMemoryPromptBuilder(
        personal_memory_store,
        config=PersonalMemoryPromptConfig(
            max_bytes=1000000,
            max_topic_files=15,
        ),
    )
    confirmation_gate = ConfirmationGate(
        journal=journal_store,
        store=personal_memory_store,
        state_path=personal_memory_dir(user_id) / "confirmation_state.json",
    )
    
    yield {
        "user_id": user_id,
        "personal_memory_store": personal_memory_store,
        "journal_store": journal_store,
        "task_history_store": task_history_store,
        "retrieval_broker": retrieval_broker,
        "session_store": session_store,
        "memory_store": memory_store,
        "personal_memory_prompt": personal_memory_prompt,
        "confirmation_gate": confirmation_gate,
        "rag_system": rag_system,
    }

@pytest.mark.parametrize("prompt_obj", OFFLINE_PROMPTS, ids=lambda x: x["id"])
def test_offline_agent_eval(offline_state, prompt_obj):
    agents_mgr.rebuild({})
    mock_ws = MockWebSocket()
    data = {"content": prompt_obj["prompt"]}
    
    response_text = ""
    tool_calls_observed = []
    
    import asyncio
    async def run():
        async with httpx.AsyncClient() as client:
            state = SharedState(
                http_client=client,
                session_store=offline_state["session_store"],
                memory_store=offline_state["memory_store"],
                personal_memory_store=offline_state["personal_memory_store"],
                personal_memory_prompt=offline_state["personal_memory_prompt"],
                journal_store=offline_state["journal_store"],
                confirmation_gate=offline_state["confirmation_gate"],
                task_history_store=offline_state["task_history_store"],
                rag_system=offline_state["rag_system"],
                retrieval_broker=offline_state["retrieval_broker"],
                user_id=offline_state["user_id"],
            )
            # Run the agent pipeline offline directly
            result = await run_agent_with_fallbacks(
                agents_mgr.main_assistant,
                agents_mgr.main_assistant_fallbacks,
                prompt_obj["prompt"],
                deps=state
            )
            return result
            
    result = asyncio.run(run())
    response_text = result.data
    
    for msg in result.new_messages():
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_calls_observed.append(part.tool_name)
                
    expected = set(prompt_obj.get("expected_tool_calls", []))
    observed = set(tool_calls_observed)
    
    # Basic scoring logic
    if not expected:
        tool_accuracy = 1.0 if not observed else 0.0
    else:
        tp = len(expected & observed)
        precision = tp / len(observed) if observed else 0.0
        recall = tp / len(expected)
        tool_accuracy = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        
    has_citation = (
        "http://" in response_text or
        "https://" in response_text or
        "source:" in response_text.lower() or
        "according to" in response_text.lower()
    ) if response_text else False

    hallucination_risk = (
        not has_citation and
        bool(expected & {"search_web", "search_url"}) and
        bool(response_text)
    )
    
    assert tool_accuracy >= 0.8, f"Tool accuracy too low. Expected {expected}, got {observed}"
    assert not hallucination_risk, f"Hallucination risk detected! Response: {response_text}"
