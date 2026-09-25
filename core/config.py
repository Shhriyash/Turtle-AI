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

from pydantic import Field, SecretStr, field_validator
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
    # Anchor the data dir to the repo root, NOT the CWD. A bare Path("data") is
    # CWD-relative, so launching the server from another directory silently
    # created a fresh empty data/users.sqlite next to that CWD — and the next
    # onboarding minted brand-new user_ids while the real memory sat orphaned
    # under <repo>/data (hazard H2 in the W2 identity audit). _PROJECT_ROOT/"data"
    # matches core.paths.DATA_DIR so identity and memory always agree on location.
    # The TURTLE_DATA_DIR env alias still overrides this for real relocations.
    data_dir: Path = Field(default=_PROJECT_ROOT / "data", alias="TURTLE_DATA_DIR")

    @field_validator("data_dir")
    @classmethod
    def _anchor_data_dir(cls, value: Path) -> Path:
        # A RELATIVE env override (TURTLE_DATA_DIR=data) would reintroduce the
        # exact CWD hazard the absolute default fixes, so anchor it to the repo
        # root too. Absolute overrides pass through untouched.
        if not value.is_absolute():
            return _PROJECT_ROOT / value
        return value

    port: int = Field(default=8765, alias="TURTLE_PORT")
    host: str = Field(default="127.0.0.1", alias="TURTLE_HOST")
    server_reload: bool = Field(default=False, alias="TURTLE_SERVER_RELOAD")

    # -----------------------------------------------------------------------
    # Cloud backends (TURTLE_DEPLOY=cloud). All optional so local mode is
    # unaffected; the cloud stores validate their own presence at startup.
    # -----------------------------------------------------------------------
    # Neon Postgres connection string. Vercel's Neon integration injects several
    # aliases (DATABASE_URL, POSTGRES_URL, ...); DATABASE_URL is canonical here.
    # Use the POOLED endpoint (…-pooler.…) on serverless so ephemeral invocations
    # don't exhaust direct connections.
    database_url: Optional[SecretStr] = Field(default=None, alias="DATABASE_URL")
    # Upstash Redis. Accept the standard rediss:// URL (REDIS_URL) or Upstash's
    # own alias (UPSTASH_REDIS_URL); redis_url resolves whichever is set.
    redis_url_primary: Optional[SecretStr] = Field(default=None, alias="REDIS_URL")
    redis_url_upstash: Optional[SecretStr] = Field(default=None, alias="UPSTASH_REDIS_URL")
    # RETIRED (WP 1.B / S-7.3): used to be the ONE secret both GitHub Actions'
    # cron-tick call AND Turtle's own self-invoked job calls (Discord
    # deferred processing, the embed-personal-memory self-invoke)
    # authenticated with — one leaked value bought an attacker both
    # surfaces, and receiving endpoints trusted identity fields straight out
    # of the body. Split below into cron_tick_secret (GitHub Actions only)
    # and internal_job_secret (Turtle-to-Turtle self-calls only, paired with
    # an HMAC envelope — see core/internal_auth.py). Nothing reads this field
    # any more; left declared (not deleted) so a running instance that still
    # has CRON_SHARED_SECRET set in its environment doesn't fail to boot.
    # Safe for the owner to remove from the environment (and eventually this
    # field) once the split has deployed everywhere.
    cron_shared_secret: Optional[SecretStr] = Field(default=None, alias="CRON_SHARED_SECRET")
    # The ONLY secret GitHub Actions holds — presented as a bearer token by
    # .github/workflows/cron-tick.yml to POST /internal/cron-tick
    # (apps/cron_tick_routes.py). Scoped to that one endpoint; cannot be used
    # to authenticate a Turtle-to-Turtle job self-call (see
    # internal_job_secret below). No default: the endpoint refuses to run
    # without it in cloud.
    cron_tick_secret: Optional[SecretStr] = Field(default=None, alias="CRON_TICK_SECRET")
    # NEVER leaves Vercel — authenticates Turtle's own self-invoked internal
    # job calls: apps/channels/discord.py's deferred-interaction self-invoke
    # (POST /channels/discord/process) and core/worker.py's
    # embed-personal-memory self-invoke (POST /internal/embed-personal-memory).
    # Used both as a bearer token AND as the HMAC key for the signed-request
    # envelope (core/internal_auth.py) those two endpoints require. GitHub
    # Actions never sees this value.
    internal_job_secret: Optional[SecretStr] = Field(default=None, alias="INTERNAL_JOB_SECRET")
    # Secret token verified on the Telegram webhook (Phase 4), sent by Telegram
    # in the X-Telegram-Bot-Api-Secret-Token header (set via setWebhook).
    telegram_webhook_secret: Optional[SecretStr] = Field(
        default=None, alias="TELEGRAM_WEBHOOK_SECRET"
    )

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

    # Channel Adapters — Discord (registered BOT application; never a self-bot)
    discord_bot_token: Optional[SecretStr] = Field(default=None, alias="DISCORD_BOT_TOKEN")
    discord_public_key: Optional[SecretStr] = Field(default=None, alias="DISCORD_PUBLIC_KEY")
    discord_application_id: Optional[str] = Field(default=None, alias="DISCORD_APPLICATION_ID")

    # Channel Adapters — Telegram (registered BOT via @BotFather; never a user
    # account). api_id / api_hash from my.telegram.org are NOT needed for the
    # HTTP Bot API — python-telegram-bot only needs the bot token.
    telegram_bot_token: Optional[SecretStr] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")

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

    # Google Maps Platform — Places API (New) + Routes API.
    # A single API key covers both surfaces. Enable the "Places API (New)" and
    # "Routes API" in the Google Cloud project and restrict the key to those
    # APIs plus your server's IP or HTTP referrer.
    google_maps_api_key: Optional[SecretStr] = Field(
        default=None, alias="GOOGLE_MAPS_API_KEY"
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
    # would push the directory past this cap raise StorageCapExceededError; the
    # write is blocked (non-destructive — nothing already stored is touched) and
    # the user is notified via a WebSocket "notice" frame (code "storage_cap")
    # that the browser renders as a toast, so failed memory writes are visible
    # instead of silent. 0 disables the cap.
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
    personal_memory_stage_b_enabled: bool = Field(
        default=True, alias="TURTLE_PERSONAL_MEMORY_STAGE_B_ENABLED"
    )
    personal_memory_stage_b_model: str = Field(
        default="openai/gpt-oss-120b", alias="TURTLE_PERSONAL_MEMORY_STAGE_B_MODEL"
    )
    episodic_summary_model: str = Field(
        default="openai/gpt-oss-20b", alias="TURTLE_EPISODIC_SUMMARY_MODEL"
    )
    personal_memory_stage_b_max_turns: int = Field(
        default=60, alias="TURTLE_PERSONAL_MEMORY_STAGE_B_MAX_TURNS"
    )
    personal_memory_stage_b_max_candidates: int = Field(
        default=8, alias="TURTLE_PERSONAL_MEMORY_STAGE_B_MAX_CANDIDATES"
    )
    # Per-turn LLM extractor (Stage A2). Was llama-3.3-70b-versatile, which Groq
    # decommissioned (404 every turn — silently swallowed). Repointed to the
    # roster Groq model, gpt-oss-20b.
    personal_memory_turn_extractor_model: str = Field(
        default="openai/gpt-oss-20b", alias="TURTLE_PERSONAL_MEMORY_TURN_EXTRACTOR_MODEL"
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

    @property
    def redis_url(self) -> Optional[str]:
        """Resolved Redis connection URL, whichever env alias is set.

        REDIS_URL wins when both are present (an explicit override beats the
        Marketplace-injected default).
        """
        for secret in (self.redis_url_primary, self.redis_url_upstash):
            if secret is not None:
                value = secret.get_secret_value().strip()
                if value:
                    return value
        return None


# Global singleton
settings = TurtleSettings()
