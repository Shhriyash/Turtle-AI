"""
tools/contracts.py
------------------
Typed result envelope for every Turtle tool.

Every tool wrapper must return ToolResult[T] so the graph executor,
fallback strategy, and hallucination validator all operate on a
predictable shape — never on stringified exceptions.

Status codes:
  ok              — call succeeded, data is populated
  invalid         — caller-supplied args are malformed (do NOT retry blindly)
  rate_limited    — upstream rate-limit; retry after retry_after_ms
  upstream_error  — upstream 5xx / transient network failure; retryable
  empty           — call succeeded but found nothing (e.g. zero search hits)
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

ToolStatus = Literal["ok", "invalid", "rate_limited", "upstream_error", "empty"]


class ToolResult(BaseModel, Generic[T]):
    """Universal return envelope for all Turtle tools."""

    status: ToolStatus
    data: T | None = None
    error_code: str = ""
    error_message: str = ""
    retryable: bool = False
    retry_after_ms: int = 0

    # ------------------------------------------------------------------ #
    # Convenience constructors                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def ok(cls, data: T) -> "ToolResult[T]":
        return cls(status="ok", data=data)

    @classmethod
    def empty(cls, message: str = "") -> "ToolResult[T]":
        return cls(status="empty", error_message=message)

    @classmethod
    def invalid(cls, message: str, code: str = "invalid_args") -> "ToolResult[T]":
        return cls(status="invalid", error_code=code, error_message=message, retryable=False)

    @classmethod
    def rate_limited(cls, retry_after_ms: int = 60_000, message: str = "") -> "ToolResult[T]":
        return cls(
            status="rate_limited",
            error_code="rate_limited",
            error_message=message,
            retryable=True,
            retry_after_ms=retry_after_ms,
        )

    @classmethod
    def upstream_error(cls, message: str, code: str = "upstream_error", retryable: bool = True) -> "ToolResult[T]":
        return cls(
            status="upstream_error",
            error_code=code,
            error_message=message,
            retryable=retryable,
        )

    # ------------------------------------------------------------------ #
    # Helpers for agent consumption                                        #
    # ------------------------------------------------------------------ #

    def to_agent_string(self) -> str:
        """Render to a terse string the LLM sees as a tool return value."""
        if self.status == "ok" and self.data is not None:
            if isinstance(self.data, str):
                return self.data
            if isinstance(self.data, BaseModel):
                return self.data.model_dump_json()
            return str(self.data)
        if self.status == "empty":
            return self.error_message or "No results found."
        if self.status == "invalid":
            return f"[Tool error — invalid args] {self.error_message}"
        if self.status == "rate_limited":
            return f"[Tool error — rate limited] Retry after {self.retry_after_ms}ms."
        if self.status == "upstream_error":
            return f"[Tool error — upstream failure] {self.error_message}"
        return f"[Tool error] {self.error_message}"


# ------------------------------------------------------------------ #
# Typed arg BaseModels for each tool (used by pydantic-ai registration)
# ------------------------------------------------------------------ #


class WebSearchArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=300,
        description=(
            "The search query. Be specific. Do NOT send raw user text verbatim; "
            "extract the core information need (e.g. 'Tokyo time now' not "
            "'what time is it in Tokyo right now please'). "
            "Use English unless the user explicitly asked for another language."
        ),
    )


class UrlFetchArgs(BaseModel):
    url: str = Field(
        description=(
            "Fully-qualified URL to fetch, including scheme (https://). "
            "Must be a real URL present in the conversation or a tool result — "
            "never invented."
        )
    )


class EmailArgs(BaseModel):
    query: str = Field(
        description=(
            "The user's email request in their own words. "
            "Include recipients, subject, body, and any cc/bcc mentions. "
            "Do NOT paraphrase or summarise — pass through relevant raw user text."
        )
    )


class HistoryArgs(BaseModel):
    query: str = Field(
        min_length=2,
        description=(
            "Natural-language question about prior conversations or remembered facts. "
            "E.g. 'What did we discuss about the Finae project last week?'"
        ),
    )


class RecallArgs(BaseModel):
    query: str = Field(
        min_length=2,
        description=(
            "Natural-language question about past context, preferences, or tasks. "
            "Use the user's words; do not paraphrase." 
        ),
    )
    scope: Literal["personal", "episodic", "tasks", "working"] = Field(
        description=(
            "Recall scope: personal (profile/journal), episodic (RAG), "
            "tasks (tool history), working (earlier in current chat)."
        )
    )


class CalendarCreateArgs(BaseModel):
    title: str = Field(description="Event title / summary.")
    start_iso: str = Field(
        description=(
            "Start datetime in ISO 8601 format with timezone offset, "
            "e.g. '2026-05-10T14:00:00+05:30'. Derive from the user's explicit statement only."
        )
    )
    end_iso: str = Field(description="End datetime ISO 8601. Must be after start_iso.")
    attendee_emails: list[str] = Field(
        default_factory=list,
        description="Attendee email addresses. Only include emails the user explicitly stated.",
    )
    description: str = Field(default="", description="Optional event description / agenda.")
    add_google_meet: bool = Field(default=True, description="Attach a Google Meet link.")


class CalendarListArgs(BaseModel):
    max_results: int = Field(default=5, ge=1, le=20, description="Number of upcoming events to return.")
    time_min_iso: str = Field(
        default="",
        description="Only return events after this ISO 8601 datetime. Defaults to now if empty.",
    )
