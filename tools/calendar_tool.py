"""
tools/calendar_tool.py
-----------------------
F4: Google Calendar API tool — graph node for the calendar intent.

Capabilities:
  - create_calendar_event(): create an event with attendees, returns Meet link + event URL
  - list_upcoming_events(): list next N events on the primary calendar

Auth:
  Two modes depending on what credentials are configured:
  1. OAuth2 user token (GOOGLE_CALENDAR_TOKEN_JSON) — for personal accounts
  2. Service account (GOOGLE_CALENDAR_CREDENTIALS_JSON with type=service_account) — for
     workspace deployments

  Both credential formats are JSON; the code detects which to use from the "type" field.

Required env vars (at least one credential source):
  GOOGLE_CALENDAR_CREDENTIALS_JSON   — raw JSON string (OAuth2 client or service account)
  GOOGLE_CALENDAR_TOKEN_JSON         — raw JSON string (OAuth2 token, for user-authorized flow)

Tool args schema is registered in tools/contracts.py (CalendarArgs).
Returns ToolResult[CalendarEventResult].

When to use (agent-facing docstring, loaded by tool registration):
  - User asks to "schedule a meeting", "create an event", "book a call", "find free time"
  - User asks "what's on my calendar", "do I have anything this week"
  - NEVER invent attendee emails — only use emails explicitly stated by the user
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from core.config import settings
from tools.contracts import ToolResult


# Google Calendar's `start.timeZone` is a hint used to render the event and to
# resolve wall-clock times when the datetime string carries NO offset. When the
# ISO string DOES carry an offset (`+05:30`, `Z`, ...) the instant is already
# unambiguous — but a naive `T14:00:00` with `timeZone=UTC` would silently
# shift a 2 PM IST meeting to 7:30 PM IST. We honour an explicit offset when
# present and fall back to the user's business timezone otherwise.
_ISO_OFFSET_RE = re.compile(r"([+-]\d{2}:?\d{2}|Z)$")

# Default when the LLM emits a naive datetime and we have no better signal.
# For this deployment the operator lives in India; override via env if needed.
_DEFAULT_TZ = "Asia/Kolkata"


def _timezone_for(iso_string: str) -> str:
    """Pick the timeZone hint to send to Google Calendar for an ISO datetime.

    - Explicit offset (`+05:30`, `-08:00`, `Z`) → 'UTC' (the offset already
      pins the instant; the tz hint only controls display, and UTC is safe).
    - Naive (`2026-05-10T14:00:00`) → fall back to the operator's local zone,
      so the LLM saying "2 PM" lands at 2 PM local rather than 7:30 PM local.
    """
    if _ISO_OFFSET_RE.search(iso_string or ""):
        return "UTC"
    return _DEFAULT_TZ


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class CalendarEventResult(BaseModel):
    event_id: str
    title: str
    start: str          # ISO 8601
    end: str            # ISO 8601
    html_link: str      # Google Calendar event URL
    meet_link: str = "" # Google Meet URL if conference data attached
    attendees: list[str] = Field(default_factory=list)


class CalendarEventListResult(BaseModel):
    events: list[CalendarEventResult]


# ---------------------------------------------------------------------------
# Args schemas (also declared in contracts.py for pydantic-ai registration)
# ---------------------------------------------------------------------------

class CalendarCreateArgs(BaseModel):
    title: str = Field(description="Event title / summary.")
    start_iso: str = Field(
        description=(
            "Start datetime in ISO 8601 format with timezone offset, "
            "e.g. '2026-05-10T14:00:00+05:30'. NEVER invent dates — derive "
            "from the user's explicit statement."
        )
    )
    end_iso: str = Field(
        description="End datetime in ISO 8601 format. Must be after start_iso."
    )
    attendee_emails: list[str] = Field(
        default_factory=list,
        description=(
            "Email addresses of attendees. Only include emails the user explicitly stated. "
            "Do NOT guess or fabricate emails."
        ),
    )
    description: str = Field(default="", description="Optional event description / agenda.")
    add_google_meet: bool = Field(
        default=True,
        description="If True, attach a Google Meet link to the event.",
    )
    notify_attendees: bool = Field(
        default=False,
        description=(
            "Send calendar invite emails to attendees. Set True ONLY when "
            "the user explicitly asked to notify/invite/email the "
            "attendees. Defaults to False — attendees are added to the "
            "event silently, with no notification sent, even when "
            "attendee_emails is non-empty."
        ),
    )


class CalendarListArgs(BaseModel):
    max_results: int = Field(default=5, ge=1, le=20, description="Number of upcoming events to return.")
    time_min_iso: Optional[str] = Field(
        default=None,
        description="Only return events starting after this ISO 8601 datetime. Defaults to now.",
    )


def render_calendar_draft(args: CalendarCreateArgs) -> str:
    """Render a proposed event legibly enough that a user can spot a wrong
    date or a wrong attendee before confirming — the entire point of the
    calendar_create/calendar_confirm draft step."""
    lines = [
        "Here's the event I'll create once you confirm:",
        f"Title: {args.title}",
        f"Start: {args.start_iso}",
        f"End: {args.end_iso}",
    ]
    lines.append(
        f"Attendees: {', '.join(args.attendee_emails)}" if args.attendee_emails else "Attendees: (none)"
    )
    if args.description:
        lines.append(f"Description: {args.description}")
    lines.append(f"Google Meet: {'yes' if args.add_google_meet else 'no'}")
    lines.append(f"Notify attendees by email: {'yes' if args.notify_attendees else 'no'}")
    lines.append('\nSay "confirm" to create it, or tell me what to change.')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def _load_token_json(user_id: Optional[str]) -> Optional[str]:
    """Resolve the OAuth2 user token JSON to use.

    Prefers the per-user token written by the in-app connect flow
    (apps/calendar_oauth_routes.py, GET /integrations/google_calendar/connect):
    Postgres in cloud mode (TURTLE_DEPLOY=cloud — the local disk path below
    does not survive a serverless cold start), personal_memory_dir(user_id)/
    google_calendar_token.json locally. Either way, each signed-in user's
    calendar_create/calendar_list calls act on THEIR OWN calendar.

    Falls back to the legacy global GOOGLE_CALENDAR_TOKEN_JSON env var for
    single-tenant / dev deployments that predate the per-user connect flow, or
    when no user_id is available (e.g. a non-web channel not yet resolved to
    a per-user token).

    Safe to call synchronously here: every caller in this module reaches
    _load_token_json via asyncio.to_thread (see create_calendar_event/
    list_calendar_events's _sync_* helpers), so the Postgres read below
    (psycopg, sync) never blocks the event loop.
    """
    if user_id:
        stored: Optional[str] = None
        if settings.is_cloud:
            try:
                from core.storage.cloud.calendar_token_store import get_token_json

                stored = get_token_json(user_id)
            except Exception:
                stored = None  # fall through to the legacy env var
        else:
            try:
                from core.paths import personal_memory_dir
                token_path = personal_memory_dir(user_id) / "google_calendar_token.json"
                if token_path.exists():
                    stored = token_path.read_text(encoding="utf-8")
            except Exception:
                stored = None  # fall through to the legacy env var
        if stored:
            # Transparently decrypts a key_version>=1 envelope, or returns a
            # pre-encryption plaintext blob unchanged — see
            # core/calendar_token_crypto.py and
            # apps/calendar_oauth_routes.py's _read_token, which this mirrors
            # (the connect-flow route module owns writing; this tool only
            # ever reads).
            try:
                from core.calendar_token_crypto import decrypt_stored, parse_key

                key_secret = settings.calendar_token_key
                key = parse_key(key_secret.get_secret_value()) if key_secret is not None else None
                token_json, _key_version = decrypt_stored(stored, key)
                return token_json
            except Exception:
                pass  # fall through to the legacy env var
    return settings.google_calendar_token_json


def _load_credentials(user_id: Optional[str] = None):
    """
    Load Google Calendar credentials from env config plus, when available,
    a per-user OAuth token on disk (see _load_token_json).
    Returns a google.oauth2.credentials.Credentials or
    google.oauth2.service_account.Credentials object, or None if unconfigured.
    """
    creds_json = settings.google_calendar_credentials_json

    if not creds_json:
        return None

    try:
        creds_data = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GOOGLE_CALENDAR_CREDENTIALS_JSON is not valid JSON: {exc}") from exc

    cred_type = creds_data.get("type", "")

    if cred_type == "service_account":
        from google.oauth2 import service_account  # type: ignore[import]
        # Narrowed from the full https://www.googleapis.com/auth/calendar to
        # event-level access (WP1.E2 / ledger 1b.3, matching the OAuth scope
        # narrowing in apps/calendar_oauth_routes.py) — this module only ever
        # calls events().insert/events().list on calendarId="primary" below,
        # never calendars().list/insert/delete.
        return service_account.Credentials.from_service_account_info(
            creds_data,
            scopes=["https://www.googleapis.com/auth/calendar.events"],
        )

    # OAuth2 client credentials + user token
    token_json = _load_token_json(user_id)
    if token_json:
        from google.oauth2.credentials import Credentials  # type: ignore[import]
        token_data = json.loads(token_json)
        client_block = creds_data.get("installed", creds_data.get("web", creds_data))
        return Credentials(
            # Deliberately omit the short-lived access token (Google's token
            # endpoint calls it "access_token"; legacy manual configs used
            # "token") and leave it unset instead. With no access token,
            # google-auth's Credentials.valid is False, which forces a
            # refresh via refresh_token on first use every time — simpler and
            # more robust than tracking each token's real expiry ourselves.
            token=None,
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=client_block.get("client_id"),
            client_secret=client_block.get("client_secret"),
        )

    return None


def _build_service(user_id: Optional[str] = None):
    """Build the Google Calendar API service client."""
    from googleapiclient.discovery import build  # type: ignore[import]
    creds = _load_credentials(user_id)
    if creds is None:
        raise RuntimeError(
            "Google Calendar credentials not configured. Set "
            "GOOGLE_CALENDAR_CREDENTIALS_JSON, then connect a calendar via "
            "GET /integrations/google_calendar/connect (or set the legacy "
            "GOOGLE_CALENDAR_TOKEN_JSON for a single-tenant deployment)."
        )
    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def create_calendar_event(
    args: CalendarCreateArgs, *, user_id: Optional[str] = None
) -> ToolResult[CalendarEventResult]:
    """
    Create a Google Calendar event and optionally attach a Google Meet link.
    Returns ToolResult[CalendarEventResult] with the event URL and Meet link.

    ``user_id``, when given, selects that user's own connected calendar (see
    apps/calendar_oauth_routes.py); omit it to use the legacy single-tenant
    GOOGLE_CALENDAR_TOKEN_JSON env var.
    """
    import asyncio

    if not settings.google_calendar_credentials_json:
        return ToolResult.invalid(
            "Google Calendar credentials not configured. Set GOOGLE_CALENDAR_CREDENTIALS_JSON.",
            code="credentials_missing",
        )

    def _sync_create() -> CalendarEventResult:
        service = _build_service(user_id)

        body: dict = {
            "summary": args.title,
            "description": args.description,
            "start": {"dateTime": args.start_iso, "timeZone": _timezone_for(args.start_iso)},
            "end": {"dateTime": args.end_iso, "timeZone": _timezone_for(args.end_iso)},
        }

        if args.attendee_emails:
            body["attendees"] = [{"email": e} for e in args.attendee_emails]

        if args.add_google_meet:
            import uuid
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": uuid.uuid4().hex,
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }

        created = service.events().insert(
            calendarId="primary",
            body=body,
            conferenceDataVersion=1 if args.add_google_meet else 0,
            # "all" only when the user explicitly asked to notify attendees —
            # never inferred merely from attendee_emails being non-empty.
            sendUpdates="all" if args.notify_attendees else "none",
        ).execute()

        meet_link = ""
        conf = created.get("conferenceData", {})
        for ep in conf.get("entryPoints", []):
            if ep.get("entryPointType") == "video":
                meet_link = ep.get("uri", "")
                break

        return CalendarEventResult(
            event_id=created["id"],
            title=created.get("summary", args.title),
            start=created["start"].get("dateTime", created["start"].get("date", "")),
            end=created["end"].get("dateTime", created["end"].get("date", "")),
            html_link=created.get("htmlLink", ""),
            meet_link=meet_link,
            attendees=[a["email"] for a in created.get("attendees", [])],
        )

    try:
        result = await asyncio.to_thread(_sync_create)
        return ToolResult.ok(result)
    except RuntimeError as exc:
        return ToolResult.invalid(str(exc), code="credentials_missing")
    except Exception as exc:
        return ToolResult.upstream_error(str(exc))


async def list_upcoming_events(
    args: CalendarListArgs, *, user_id: Optional[str] = None
) -> ToolResult[CalendarEventListResult]:
    """List upcoming events from the user's primary Google Calendar.

    ``user_id``, when given, selects that user's own connected calendar (see
    apps/calendar_oauth_routes.py); omit it to use the legacy single-tenant
    GOOGLE_CALENDAR_TOKEN_JSON env var.
    """
    import asyncio

    if not settings.google_calendar_credentials_json:
        return ToolResult.invalid(
            "Google Calendar credentials not configured. Set GOOGLE_CALENDAR_CREDENTIALS_JSON.",
            code="credentials_missing",
        )

    def _sync_list() -> CalendarEventListResult:
        service = _build_service(user_id)
        now = args.time_min_iso or datetime.now(timezone.utc).isoformat()

        items = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=args.max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute().get("items", [])

        events = []
        for item in items:
            start = item["start"].get("dateTime", item["start"].get("date", ""))
            end = item["end"].get("dateTime", item["end"].get("date", ""))
            conf = item.get("conferenceData", {})
            meet_link = ""
            for ep in conf.get("entryPoints", []):
                if ep.get("entryPointType") == "video":
                    meet_link = ep.get("uri", "")
                    break
            events.append(CalendarEventResult(
                event_id=item["id"],
                title=item.get("summary", "(No title)"),
                start=start,
                end=end,
                html_link=item.get("htmlLink", ""),
                meet_link=meet_link,
                attendees=[a["email"] for a in item.get("attendees", [])],
            ))

        return CalendarEventListResult(events=events)

    try:
        result = await asyncio.to_thread(_sync_list)
        return ToolResult.ok(result)
    except RuntimeError as exc:
        return ToolResult.invalid(str(exc), code="credentials_missing")
    except Exception as exc:
        return ToolResult.upstream_error(str(exc))
