Recall tool

Use this tool to retrieve prior context beyond the visible prompt.

Arguments:
- query: natural-language search query describing what you need.
- scope: one of personal, episodic, tasks, working.

Scopes:
- personal: personal memory topics and journal events (identity, preferences, workflow, contacts, projects). This is the authoritative store for anything the user has told Turtle before (facts, preferences, contacts, relations, projects).
- episodic: past conversation summaries stored in RAG. Use scope="episodic" for questions about something discussed in a PREVIOUS conversation — "What did we discuss about X?", "Last time we talked about Y, what did we decide?", "Remind me what we covered before". It reflects what was said in past sessions; it does not update mid-conversation.
- tasks: tool history records (searches, emails, calendar actions).
- working: earlier parts of the current conversation beyond the visible window.

Rules:
- Use the user's words in the query; do not paraphrase.
- Call recall only when needed; do not use it for general knowledge.
- Do not use recall for current events, news, live facts, prices, or schedules — use search_web for those.
- If recall returns no relevant info, say so and ask a follow-up question if needed.
