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


class TurtleSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    # -----------------------------------------------------------------------
    # Memory Subsystem
    # -----------------------------------------------------------------------
    personal_memory_enabled: bool = Field(
        default=True, alias="TURTLE_PERSONAL_MEMORY_ENABLED"
    )
    personal_memory_dream_pass_enabled: bool = Field(
        default=True, alias="TURTLE_PERSONAL_MEMORY_DREAM_PASS_ENABLED"
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
    personal_memory_max_bytes: int = Field(
        default=1024 * 1024, alias="TURTLE_PERSONAL_MEMORY_MAX_BYTES"
    )
    personal_memory_max_topic_files: int = Field(
        default=15, alias="TURTLE_PERSONAL_MEMORY_MAX_TOPIC_FILES"
    )

    # Periodic reflector (Phase 2): runs Stage B + dream pass mid-session.
    reflect_enabled: bool = Field(default=True, alias="TURTLE_REFLECT_ENABLED")
    reflect_every_turns: int = Field(default=15, alias="TURTLE_REFLECT_EVERY_TURNS")
    reflect_idle_seconds: int = Field(default=1800, alias="TURTLE_REFLECT_IDLE_SECONDS")
    reflect_max_consecutive_failures: int = Field(
        default=3, alias="TURTLE_REFLECT_MAX_CONSECUTIVE_FAILURES"
    )

    @property
    def is_cloud(self) -> bool:
        """Returns True if running in cloud deployment mode (enables Arq, strict auth, etc)."""
        return self.deploy_mode.lower() == "cloud"


# Global singleton
settings = TurtleSettings()
