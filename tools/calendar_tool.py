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
from datetime import datetime, timezone, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from core.config import settings
from tools.contracts import ToolResult


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


class CalendarListArgs(BaseModel):
    max_results: int = Field(default=5, ge=1, le=20, description="Number of upcoming events to return.")
    time_min_iso: Optional[str] = Field(
        default=None,
        description="Only return events starting after this ISO 8601 datetime. Defaults to now.",
    )


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def _load_credentials():
    """
    Load Google Calendar credentials from env config.
    Returns a google.oauth2.credentials.Credentials or
    google.oauth2.service_account.Credentials object, or None if unconfigured.
    """
    creds_json = settings.google_calendar_credentials_json
    token_json = settings.google_calendar_token_json

    if not creds_json:
        return None

    try:
        creds_data = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GOOGLE_CALENDAR_CREDENTIALS_JSON is not valid JSON: {exc}") from exc

    cred_type = creds_data.get("type", "")

    if cred_type == "service_account":
        from google.oauth2 import service_account  # type: ignore[import]
        return service_account.Credentials.from_service_account_info(
            creds_data,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )

    # OAuth2 client credentials + user token
    if token_json:
        from google.oauth2.credentials import Credentials  # type: ignore[import]
        token_data = json.loads(token_json)
        return Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=creds_data.get("installed", creds_data.get("web", creds_data)).get("client_id"),
            client_secret=creds_data.get("installed", creds_data.get("web", creds_data)).get("client_secret"),
        )

    return None


def _build_service():
    """Build the Google Calendar API service client."""
    from googleapiclient.discovery import build  # type: ignore[import]
    creds = _load_credentials()
    if creds is None:
        raise RuntimeError(
            "Google Calendar credentials not configured. "
            "Set GOOGLE_CALENDAR_CREDENTIALS_JSON (and GOOGLE_CALENDAR_TOKEN_JSON for OAuth2)."
        )
    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def create_calendar_event(args: CalendarCreateArgs) -> ToolResult[CalendarEventResult]:
    """
    Create a Google Calendar event and optionally attach a Google Meet link.
    Returns ToolResult[CalendarEventResult] with the event URL and Meet link.
    """
    import asyncio

    if not settings.google_calendar_credentials_json:
        return ToolResult.invalid(
            "Google Calendar credentials not configured. Set GOOGLE_CALENDAR_CREDENTIALS_JSON.",
            code="credentials_missing",
        )

    def _sync_create() -> CalendarEventResult:
        service = _build_service()

        body: dict = {
            "summary": args.title,
            "description": args.description,
            "start": {"dateTime": args.start_iso, "timeZone": "UTC"},
            "end": {"dateTime": args.end_iso, "timeZone": "UTC"},
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
            sendUpdates="all" if args.attendee_emails else "none",
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


async def list_upcoming_events(args: CalendarListArgs) -> ToolResult[CalendarEventListResult]:
    """List upcoming events from the user's primary Google Calendar."""
    import asyncio

    if not settings.google_calendar_credentials_json:
        return ToolResult.invalid(
            "Google Calendar credentials not configured. Set GOOGLE_CALENDAR_CREDENTIALS_JSON.",
            code="credentials_missing",
        )

    def _sync_list() -> CalendarEventListResult:
        service = _build_service()
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
