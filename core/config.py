"""
core/config.py
--------------
G6: Centralised configuration using pydantic-settings.

Replaces ad-hoc os.getenv() calls across the codebase with a single validated
Settings object loaded from the environment (or .env file).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class TurtleSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Deployment & Infrastructure
    # -----------------------------------------------------------------------
    deploy_mode: str = Field(default="local", alias="TURTLE_DEPLOY")
    data_dir: Path = Field(default=Path("data"), alias="TURTLE_DATA_DIR")
    port: int = Field(default=8765, alias="TURTLE_PORT")
    host: str = Field(default="127.0.0.1", alias="TURTLE_HOST")
    server_reload: bool = Field(default=False, alias="TURTLE_SERVER_RELOAD")

    # -----------------------------------------------------------------------
    # API Keys
    # -----------------------------------------------------------------------
    openrouter_api_key: Optional[SecretStr] = Field(default=None, alias="OPENROUTER_API_KEY")
    groq_api_key: Optional[SecretStr] = Field(default=None, alias="GROQ_API_KEY")
    groq_api_key2: Optional[SecretStr] = Field(default=None, alias="GROQ_API_KEY2")
    tavily_api_key: Optional[SecretStr] = Field(default=None, alias="TAVILY_API_KEY")
    deepgram_api_key: Optional[SecretStr] = Field(default=None, alias="DEEPGRAM_API_KEY")
    logfire_token: Optional[SecretStr] = Field(default=None, alias="LOGFIRE_TOKEN")
    scraped_do_api_key: Optional[SecretStr] = Field(default=None, alias="SCRAPEDO_API_KEY")
    auth_secret_key: Optional[SecretStr] = Field(default=None, alias="AUTH_SECRET_KEY")

    # -----------------------------------------------------------------------
    # Channel Adapters — Twilio (WhatsApp + Voice)
    # -----------------------------------------------------------------------
    twilio_account_sid: Optional[SecretStr] = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[SecretStr] = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number: Optional[str] = Field(default=None, alias="TWILIO_WHATSAPP_NUMBER")
    twilio_voice_number: Optional[str] = Field(default=None, alias="TWILIO_VOICE_NUMBER")

    # Channel Adapters — Slack
    slack_bot_token: Optional[SecretStr] = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_signing_secret: Optional[SecretStr] = Field(default=None, alias="SLACK_SIGNING_SECRET")

    # Channel Adapters — SendBlue (iMessage)
    sendblue_api_key: Optional[SecretStr] = Field(default=None, alias="SENDBLUE_API_KEY")
    sendblue_api_secret: Optional[SecretStr] = Field(default=None, alias="SENDBLUE_API_SECRET")

    # Channel Adapters — Google Calendar
    google_calendar_credentials_json: Optional[str] = Field(
        default=None, alias="GOOGLE_CALENDAR_CREDENTIALS_JSON"
    )
    google_calendar_token_json: Optional[str] = Field(
        default=None, alias="GOOGLE_CALENDAR_TOKEN_JSON"
    )

    # -----------------------------------------------------------------------
    # Feature Flags & Limits
    # -----------------------------------------------------------------------
    tts_speed: float = Field(default=1.2, alias="TURTLE_TTS_SPEED")
    tts_debug: bool = Field(default=False, alias="TTS_DEBUG")
    tool_output_max_chars: int = Field(default=4000, alias="TURTLE_TOOL_OUTPUT_MAX_CHARS")

    # Branding — bot's outbound email identity (used by the email agent
    # prompt and the magic-link onboarding sender).
    bot_email: str = Field(default="iamturtleai@gmail.com", alias="TURTLE_BOT_EMAIL")

    # Dev-only escape hatch: when set, WebSocket connections without a token
    # resolve to a shared local_dev_user instead of being rejected. Must NEVER
    # be enabled in cloud / shared deployments.
    dev_anon: bool = Field(default=False, alias="TURTLE_DEV_ANON")

    # Magic-link onboarding (Phase 4)
    magic_link_jwt_ttl_minutes: int = Field(
        default=15, alias="TURTLE_MAGIC_LINK_TTL_MINUTES"
    )
    session_cookie_ttl_days: int = Field(
        default=90, alias="TURTLE_SESSION_COOKIE_TTL_DAYS"
    )
    public_base_url: str = Field(
        default="http://127.0.0.1:8765", alias="TURTLE_PUBLIC_BASE_URL"
    )
    onboarding_rate_limit_per_hour: int = Field(
        default=5, alias="TURTLE_ONBOARDING_RATE_LIMIT_PER_HOUR"
    )

    # Phase 6: production guardrails
    # Hard cap on bytes stored under personal_memory_dir(user_id). Writes that
    # would push the directory past this cap raise StorageCapExceededError so
    # the UI can prompt the user to trim. 0 disables the cap.
    user_storage_cap_mb: int = Field(
        default=50, alias="TURTLE_USER_STORAGE_CAP_MB"
    )
    # Per-user WebSocket message rate limits. 0 disables.
    ws_messages_per_hour: int = Field(
        default=60, alias="TURTLE_WS_MESSAGES_PER_HOUR"
    )
    ws_messages_per_day: int = Field(
        default=1000, alias="TURTLE_WS_MESSAGES_PER_DAY"
    )
    # Phase 7: gate /admin/* endpoints. None = endpoints return 503.
    admin_token: Optional[SecretStr] = Field(default=None, alias="TURTLE_ADMIN_TOKEN")

    # -----------------------------------------------------------------------
    # Memory Subsystem
    # -----------------------------------------------------------------------
    personal_memory_enabled: bool = Field(
        default=True, alias="TURTLE_PERSONAL_MEMORY_ENABLED"
    )
    # Phase 6: dream pass disabled by default in prod until observed on
    # 2-3 real users' journals. Re-enable explicitly via env when ready.
    personal_memory_dream_pass_enabled: bool = Field(
        default=False, alias="TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED"
    )
    personal_memory_stage_b_enabled: bool = Field(
        default=True, alias="TURTLE_PERSONAL_MEMORY_STAGE_B_ENABLED"
    )
    personal_memory_stage_b_model: str = Field(
        default="openai/gpt-oss-120b", alias="TURTLE_PERSONAL_MEMORY_STAGE_B_MODEL"
    )
    episodic_summary_model: str = Field(
        default="llama-3.1-8b-instant", alias="TURTLE_EPISODIC_SUMMARY_MODEL"
    )
    personal_memory_stage_b_max_turns: int = Field(
        default=60, alias="TURTLE_PERSONAL_MEMORY_STAGE_B_MAX_TURNS"
    )
    personal_memory_stage_b_max_candidates: int = Field(
        default=8, alias="TURTLE_PERSONAL_MEMORY_STAGE_B_MAX_CANDIDATES"
    )
    # Per-turn LLM extractor (Stage A2). 8B misclassified roles/names in
    # production; 70B is the floor for open-vocabulary extraction.
    personal_memory_turn_extractor_model: str = Field(
        default="llama-3.3-70b-versatile", alias="TURTLE_PERSONAL_MEMORY_TURN_EXTRACTOR_MODEL"
    )
    personal_memory_max_bytes: int = Field(
        default=1024 * 1024, alias="TURTLE_PERSONAL_MEMORY_MAX_BYTES"
    )
    personal_memory_max_topic_files: int = Field(
        default=15, alias="TURTLE_PERSONAL_MEMORY_MAX_TOPIC_FILES"
    )

    # Hybrid personal recall (FTS5-first, vector fallback). Tuned from logged
    # retrieval stats (§13), not gotten right up front.
    # Token-overlap fraction of the top FTS hit below which we treat lexical
    # retrieval as weak and fall back to the vector store.
    personal_recall_overlap_threshold: float = Field(
        default=0.5, alias="TURTLE_PERSONAL_RECALL_OVERLAP_THRESHOLD"
    )
    # FTS5 BM25 rank ceiling (more-negative = better). A top hit with rank
    # above this is barely-matched → treat as weak. Default 0.0 effectively
    # disables the BM25 veto (real ranks on a small per-user corpus are tiny
    # negatives), so token overlap is the deciding signal until tuned from
    # logged stats (§13). Set to e.g. -0.5 to make BM25 vote.
    personal_recall_bm25_ceiling: float = Field(
        default=0.0, alias="TURTLE_PERSONAL_RECALL_BM25_CEILING"
    )
    # Off by default: ship the simple FTS-first/vector-fallback union. Only
    # enable a normalized BM25+cosine merge once logs justify it (§12.3).
    personal_recall_merge_enabled: bool = Field(
        default=False, alias="TURTLE_PERSONAL_RECALL_MERGE_ENABLED"
    )
    # (W_LEX, W_SEM) weights for the optional normalized merge.
    personal_recall_merge_lex_weight: float = Field(
        default=0.6, alias="TURTLE_PERSONAL_RECALL_MERGE_LEX_WEIGHT"
    )
    personal_recall_merge_sem_weight: float = Field(
        default=0.4, alias="TURTLE_PERSONAL_RECALL_MERGE_SEM_WEIGHT"
    )

    # Periodic reflector (Phase 2): runs Stage B + dream pass mid-session.
    reflect_enabled: bool = Field(default=True, alias="TURTLE_REFLECT_ENABLED")
    reflect_every_turns: int = Field(default=15, alias="TURTLE_REFLECT_EVERY_TURNS")
    reflect_idle_seconds: int = Field(default=1800, alias="TURTLE_REFLECT_IDLE_SECONDS")
    reflect_max_consecutive_failures: int = Field(
        default=3, alias="TURTLE_REFLECT_MAX_CONSECUTIVE_FAILURES"
    )

    # Phase 1 / A3: cap on planner cascade size. With 3 Gemini + 3 OpenRouter
    # keys the unbounded pool reaches 7 attempts per planner call, which is
    # the dominant cost of a single news/search turn. 4 = primary + 3 fallbacks.
    planner_max_agents: int = Field(default=4, alias="TURTLE_PLANNER_MAX_AGENTS")

    # Phase 2 / B1+B2: per-turn memory extraction looks at the last N user+
    # assistant messages so multi-turn flows ("save as routine" -> "every day"
    # -> "8 am") can be parsed coherently. 6 = roughly 3 user turns + 3 replies.
    memory_extract_window_turns: int = Field(
        default=6, alias="TURTLE_MEMORY_EXTRACT_WINDOW_TURNS"
    )

    @property
    def is_cloud(self) -> bool:
        """Returns True if running in cloud deployment mode (enables Arq, strict auth, etc)."""
        return self.deploy_mode.lower() == "cloud"


# Global singleton
settings = TurtleSettings()
