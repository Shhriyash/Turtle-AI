# Turtle Issues

## Critical

- Voice mode drops short-term context. `main_assistant` is called without `message_history` in voice mode at [apps/turtle_voice.py:481](apps/turtle_voice.py:481), while text mode does pass it at [apps/turtle_voice.py:422](apps/turtle_voice.py:422). Follow-ups like `to me`, `correct the email`, `subject is...`, and `send it now` have no reliable in-session state.

- Email sending uses two LLM routing hops for a simple structured task. `main_assistant` decides whether to call `send_email_assistant`, then `email_agent` decides whether to call `send_email` at [apps/turtle_voice.py:360](apps/turtle_voice.py:360) and [apps/turtle_voice.py:261](apps/turtle_voice.py:261). This compounds STT noise and model uncertainty.

- The email prompt claims context that is not actually provided. `email_agent` says it has conversation history at [core/system_prompts/email_agent.txt:17](core/system_prompts/email_agent.txt:17), but the delegation call at [apps/turtle_voice.py:364](apps/turtle_voice.py:364) does not pass `message_history`.

- Current-session memory is not queryable. `query_history()` only searches the vector store at [rag/system/complete_rag.py:111](rag/system/complete_rag.py:111). Active session data in `current_session.json` is ignored, so the assistant cannot recall facts stated earlier in the same voice session unless they are still in `message_history`, which voice mode does not maintain.

- Session persistence is crash-unsafe. Conversations are embedded only in `end_session()` at [rag/system/complete_rag.py:150](rag/system/complete_rag.py:150). If the app crashes, the session remains only in `data/rag/current_session.json` and is not indexed. On next startup, `start_session()` overwrites that file instead of recovering it at [rag/system/complete_rag.py:53](rag/system/complete_rag.py:53).

## Memory / RAG flaws

- The storage format is lossy. You persist simplified `{user_query, turtle_response, timestamp}` records at [rag/system/complete_rag.py:82](rag/system/complete_rag.py:82) instead of native `ModelMessage` objects. This loses system prompts, tool calls, tool returns, retries, and exact turn structure.

- The chunker destroys turn boundaries. It flattens sessions into plain transcript text at [rag/chunking/json_chunking.py:64](rag/chunking/json_chunking.py:64) and then splits by character windows at [rag/chunking/json_chunking.py:101](rag/chunking/json_chunking.py:101). Stored chunks can start mid-word and mid-thought. Your own data already shows this: `t day too...` in `data/rag/vector/chunk_metadata.json`.

- Retrieved metadata is inconsistent. `query_history()` returns `timestamp` at [rag/system/complete_rag.py:136](rag/system/complete_rag.py:136), but vector metadata stores `creation_time` and `added_timestamp` at [rag/storage/vector_storage.py:123](rag/storage/vector_storage.py:123). Returned timestamps are effectively blank.

- Session timestamps are wrong on persistence. `end_session()` rewrites `creation_time` using `datetime.now()` at [rag/system/complete_rag.py:157](rag/system/complete_rag.py:157) instead of preserving the original session start time from [rag/system/complete_rag.py:64](rag/system/complete_rag.py:64).

- Retrieval quality is weak and uncalibrated. `query_history()` uses a fixed `top_k=5` and `threshold=0.3` at [rag/system/complete_rag.py:123](rag/system/complete_rag.py:123) with no recency weighting, no reranking, no query classification, and no exact-match path for structured facts like names, emails, and preferences.

- The main agent is forced to interpret raw JSON chunks. `history_tool` returns raw chunk JSON and asks the model to infer intent and extract content at [core/system_prompts/main_assistant.txt:12](core/system_prompts/main_assistant.txt:12). This is fragile and wastes tokens.

- Vector deletion is incomplete. `delete_session()` marks rows deleted but does not rebuild the FAISS index at [rag/storage/vector_storage.py:208](rag/storage/vector_storage.py:208). Deleted vectors remain in the index forever, and cleanup is explicitly a no-op at [rag/storage/vector_storage.py:231](rag/storage/vector_storage.py:231).

## Model vs goal

- The current model is capable of instruction following and tool calling, but the system asks it to do too much under noisy input. NVIDIA describes `nemotron-3-nano-30b-a3b` as strong for instruction following and tool calling, but your goal requires consistent slot filling, follow-up resolution, and tool routing under STT errors. The model is not the only problem, but the current architecture overestimates its reliability for multi-hop voice workflows.

- The free OpenRouter route adds more variability. OpenRouter explicitly routes requests across providers with fallbacks. That helps uptime, but can add behavioral variance across runs when you already depend on strict tool behavior.

- `temperature=0.2` at [apps/turtle_voice.py:178](apps/turtle_voice.py:178) is not the core bug. The larger issue is the combination of STT ambiguity, missing session history, raw JSON memory retrieval, and double delegation.

## What PydanticAI already gives you

- `message_history` is the right short-term memory mechanism. PydanticAI supports passing prior messages directly into new runs and reusing `result.new_messages()` / `result.all_messages()`. This is the missing piece in voice mode.

- Persistent session chat can be stored as native PydanticAI messages. PydanticAI documents serializing messages with `ModelMessagesTypeAdapter` and `to_jsonable_python`. That is a better session store than a custom flattened transcript.

- `history_processors` are the built-in way to control context size. Use them to keep recent turns or summarize old ones instead of replacing short-term memory with RAG.

- `args_validator` plus `ModelRetry` can make email tools more reliable. Validate recipient format, missing subject/body, and ambiguous `to me` references before the tool executes.

- `ToolReturn` can separate machine-friendly tool results from LLM-facing context. That is useful if you keep a history tool, but it is not the first fix.

- Deferred tools are useful only if you want approval or external execution for email sending. They do not solve the current routing and memory failures.

- DBOS durable execution is useful if you need crash-safe persistence and resumable workflows. It is relevant for session finalization and long-running tasks, but it is not the first fix for your current reliability issues.

## Recommended direction

- Split memory into two layers:
  - Short-term session memory: native `ModelMessage` history passed on every run.
  - Long-term memory: separate persistent facts and RAG summaries for cross-session recall.

- Stop using RAG as a substitute for active session state.

- Replace the two-agent email path with a deterministic email state machine or a single-agent flow with strong argument validation.

- Store structured facts separately from transcript chunks. Email addresses, names, preferences, and user aliases should not depend on semantic retrieval from chunked prose.

- Recover unfinished sessions on startup. If `data/rag/current_session.json` exists, ingest or restore it before starting a new session.

- If RAG remains, chunk by turn groups, preserve turn metadata, preserve exact timestamps, and add reranking or exact-match retrieval for facts.

## Useful docs

- Message history: https://ai.pydantic.dev/message-history/
- Function tools: https://ai.pydantic.dev/tools/
- Advanced tool returns and validators: https://ai.pydantic.dev/tools-advanced/
- Deferred tools: https://ai.pydantic.dev/deferred-tools/
- DBOS durable execution: https://ai.pydantic.dev/durable_execution/dbos/
- NVIDIA model page: https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b
- OpenRouter model page: https://openrouter.ai/nvidia/nemotron-3-nano-30b-a3b:free
