# Web Search Root Cause

## Symptom

The web search flow was repeatedly calling `search_web` for the same user request and often ended with:

`I’m sorry, but I’m unable to retrieve that information at the moment.`

This consumed many requests and tokens without producing a useful answer.

## Root Cause

The failure was architectural, not just prompt quality.

### 1. Weak web-search execution path

The old `search_web` tool did not perform search directly.
It delegated to a separate `web_search_agent` that relied on:

- OpenRouter
- `nvidia/nemotron-3-nano-30b-a3b:free`
- `WebSearchTool()`

That path was fragile for shopping and retrieval tasks.

### 2. Main agent had no effective loop guard

The main agent was allowed to call `search_web` repeatedly for the same user request.
When the first search result was weak or incomplete, the main agent retried the same tool instead of answering from the partial result.

### 3. Tool output quality was uncontrolled

The old tool path returned model-generated search output rather than deterministic result formatting.
That meant the main agent received inconsistent search summaries instead of stable search results.

### 4. Output formatting was not speech-safe

Tool outputs and final assistant responses could contain markdown-like formatting or punctuation patterns that sound bad in TTS.

## What Changed

### Web search

`search_web` now performs deterministic web lookup directly in Python using DuckDuckGo HTML results.

Current flow:

1. Main agent decides a web lookup is needed.
2. `search_web` runs direct search with `httpx` plus `BeautifulSoup`.
3. The tool formats the top results into plain text.
4. The cleaned plain-text result is returned to the main agent.

This removes the fragile model-mediated search execution layer.

### Loop resistance

- Search results are cached per query in active state.
- The main prompt now says `search_web` should be called at most once for the same user request unless refinement is needed.

### Output cleanup

All tool output passed back to the main agent is cleaned into plain text.
Final assistant output is also cleaned before printing and before TTS.

## Current Architecture

- `main_assistant`
  - Groq `openai/gpt-oss-120b`
- `search_web`
  - deterministic Python search
- `search_url`
  - deterministic URL extraction
- `send_email_assistant`
  - deterministic state machine plus email extraction agent
- `email_agent`
  - Groq `openai/gpt-oss-120b`

## Remaining Limitation

DuckDuckGo HTML search gives generic search results, not a structured shopping catalog.
So for product ranking requests like `top 5 gaming laptops from amazon.in`, the tool can now retrieve relevant pages reliably, but ranking quality still depends on what is publicly indexed.

If you want stronger shopping results later, the next step is a dedicated product search/scraping layer rather than another agent prompt tweak.
