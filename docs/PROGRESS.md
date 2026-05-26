# Turtle — Implementation Progress

*Last updated: 2026-05-27*

## Overall status

| Area | Phase | Status |
|------|-------|--------|
| Orchestration (circuit breaker, cap, span) | Phase 1 | ✅ Done |
| Per-turn LLM extraction + multi-turn window | Phase 2 | ✅ Done |
| Routine schema, snapshot, inline confirmation UX | Phase 3 | ✅ Done |
| APScheduler + restart-safe SQLite jobstore | Phase 4 | ✅ Done |
| Harmony-400 / empty-tool-name sanitizer | Phase 5 | ✅ Done |
| Storage cap + WS rate limiter (guardrails) | Phase 6 | ✅ Done |
| Funnel telemetry + admin routes + onboarding | Phase 7 | ✅ Done |
| Multi-user deployment (cookie identity, tenanted paths) | Phase 8 | 🔄 In progress |
| Batch /api/memory/pending review UI | Deferred | ⏳ Backlog |

---

## What shipped in this branch (`orchestration-modified`)

### Phase 6 — Production guardrails (`core/guardrails.py`)

- **`StorageCapExceededError` + `enforce_storage_cap`** — checked from
  `PersonalMemoryStore.write_topic` and `JournalStore.append`. A runaway writer
  (large paste, infinite extraction loop) cannot fill disk for all users. Cap
  driven by `TURTLE_USER_STORAGE_CAP_MB` (default 100 MB per user).
- **`WebSocketRateLimiter`** — per-user sliding-window counters (hourly +
  daily). Driven by `TURTLE_WS_MESSAGES_PER_HOUR` / `TURTLE_WS_MESSAGES_PER_DAY`.
  In-process dict + lock; swap for Redis when multi-process.

### Phase 7 — Funnel telemetry (`core/telemetry.py`)

- `emit(event, **fields)` — wraps logfire; prints to stdout when logfire
  is unavailable. Never raises.
- `emit_once(user_id, event, **fields)` — idempotency backed by a sentinel
  file at `data/memory/personal/{user_id}/.telemetry/{event}`. Returns `True`
  the first time, `False` on replay.
- Events instrumented: `onboarding_start`, `onboarding_complete`,
  `first_message_sent`, `memory_first_confirmed`, `forget_me_requested`,
  `forget_me_completed`.

### Onboarding — magic-link flow (`apps/onboarding_routes.py`)

Routes:
- `POST /onboarding/start` — validates email + name + timezone, mints a
  short-lived JWT (TTL driven by `TURTLE_MAGIC_LINK_JWT_TTL_MINUTES`), emails
  the user a claim link via the existing Gmail SMTP bot. Seeds `identity.md`
  and journals the facts immediately (so the replayer doesn't clobber them on
  first session end). In dev mode sets the session cookie without requiring
  the magic-link click.
- `GET /onboarding/claim` — verifies JWT, checks single-use `jti` via
  `identity_manager.mark_token_claimed`, seeds `identity.md`, sets
  `turtle_uid` HTTP-only cookie, redirects to `/`.

Rate-limited: max `TURTLE_ONBOARDING_RATE_LIMIT_PER_HOUR` starts per IP.

Session cookie: signed JWT carrying `{sub: user_id, channel: "web"}`.
Verify with `verify_session_cookie(token)` (also in `onboarding_routes.py`).

### Admin + GDPR routes (`apps/admin_routes.py`)

- `GET /admin/users` — lists all users with storage + RAG bytes, journal event
  count, last-seen timestamp. Requires `X-Admin-Token` header matching
  `TURTLE_ADMIN_TOKEN`. Returns 503 when the token is unset (fail-loud on
  misconfigured deploys).
- `POST /forget-me` — emails a deletion magic link to the user's primary email.
  Always returns 200 to avoid leaking email existence.
- `GET /forget-me/confirm` — verifies the deletion JWT (`kind=forget_me`),
  hard-deletes `data/memory/personal/{user_id}/`, RAG dir, and DB rows, clears
  the `turtle_uid` cookie.

### Earlier phases (already in prior commit, recorded here for reference)

**Phase 1** — `core/graph.py`: web intent bypasses planner cascade; orchestration
`logfire.span`. `core/health_tracker.py`: circuit breaker (60s/300s per failure
class). `core/llm_client.py`: cooldown skip, `_sanitize_message_history`. WS
error envelope: classified friendly messages.

**Phase 2** — async per-turn extraction (6-turn window), workflow auto-promote
on explicit confirmation.

**Phase 3** — routine value schema (`time` + `timezone`), friendly confirmation
prompt, routine aggregation in profile snapshot, workflow topic triggers.

**Phase 4** — `core/routine_scheduler.py`: `AsyncIOScheduler` +
`SQLAlchemyJobStore` at `data/scheduler.sqlite`. Scans journal for applied
`workflow.*` events; registers `CronTrigger` per routine. Fires by writing a
`workflow.scheduled_fire.<key>` event into the user's journal. Survives restarts.

**Phase 5** — `_sanitize_message_history` drops empty-`tool_name`
`ToolCallPart`s and orphaned `ToolReturnPart`s before any provider sees the
request body. Eliminates Harmony-400 deterministic failures.

---

## Key env-var additions (this branch)

| Variable | Default | Purpose |
|----------|---------|---------|
| `TURTLE_USER_STORAGE_CAP_MB` | `100` | Max MB per user in personal memory dir |
| `TURTLE_WS_MESSAGES_PER_HOUR` | `60` | WS rate limit (hourly) |
| `TURTLE_WS_MESSAGES_PER_DAY` | `500` | WS rate limit (daily) |
| `TURTLE_MAGIC_LINK_JWT_TTL_MINUTES` | `15` | Magic-link token lifetime |
| `TURTLE_SESSION_COOKIE_TTL_DAYS` | `30` | turtle_uid cookie TTL |
| `TURTLE_ONBOARDING_RATE_LIMIT_PER_HOUR` | `5` | Onboarding starts per IP/hr |
| `TURTLE_ADMIN_TOKEN` | unset | Required for /admin/* routes |
| `TURTLE_IS_CLOUD` | `false` | Cloud mode: forces magic-link click, HTTPS cookies |
| `TURTLE_PUBLIC_BASE_URL` | `http://localhost:8000` | Canonical URL for generated links |

---

## What's still to do (Phase 8 — multi-user deployment)

See `docs/MULTI_USER_DEPLOYMENT_PLAN.md` for the full breakdown. Key remaining items:

1. **Neutralise system prompt** — replace hardcoded "Shriyash's personal AI
   assistant" with a template that reads from injected identity memory at
   runtime (Phase 1 of that plan).
2. **Per-user memory paths** — `core/paths.py:personal_memory_dir` already
   accepts `user_id` but ignores it; wire it properly and audit all 23 call
   sites (`PersonalMemoryStore`, `JournalStore`, `ConfirmationGate`,
   `MemoryReplayer`, prompt builder).
3. **Cookie → user_id threading** — `turtle_server.py` WebSocket handshake
   reads `turtle_uid` cookie → `verify_session_cookie` (now in
   `onboarding_routes.py`) → `user_id` threaded through connection state.
4. **Move Shriyash's existing data** to `data/memory/personal/usr_shriyash/`
   and map the dev-machine cookie to that sub-path.
5. **Confirm two-browser isolation** with fresh cookies before sharing the URL.

---

## Deferred

- **Batch /api/memory/pending UI** — per-turn sidecar confirmation prompt
  (Phase 3) is sufficient for now. Full review panel goes in if the pending
  queue grows unmanageable.
- **Voice path** (`apps/turtle_voice.py`) — all fixes target the web/WS path.
  Same hooks exist in voice; apply the same shape as a follow-up.
- **Multi-instance scheduling** — current APScheduler runs in-process
  (single-instance). The `RoutineScheduler` class boundary makes swapping to
  Temporal (already in venv) mechanical when needed.
- **Push notifications / email delivery for fired routines** — currently,
  scheduler fires write a journal event surfaced on next connect. Email/push
  delivery is next.
