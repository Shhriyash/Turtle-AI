from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.paths import MEMORY_PROFILE_FILE, PERSONAL_MEMORY_DIR
from core.personal_memory_store import PersonalMemoryStore


@dataclass(frozen=True)
class ProfileMigrationResult:
    written_topics: list[str]
    index_entries: int


def migrate_profile_to_markdown(
    *,
    profile_path: Path,
    target_dir: Path,
    force: bool = False,
) -> ProfileMigrationResult:
    profile = _load_profile(profile_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    topic_paths = {
        "identity": target_dir / "identity.md",
        "preferences": target_dir / "preferences.md",
        "workflow": target_dir / "workflow.md",
        "contacts": target_dir / "contacts.md",
        "projects": target_dir / "projects.md",
    }

    if not force:
        existing = [path for path in topic_paths.values() if path.exists()]
        if existing or (target_dir / "MEMORY.md").exists():
            raise FileExistsError("Target personal memory directory already contains Markdown memory files. Use --force to overwrite.")

    store = PersonalMemoryStore(
        base_dir=target_dir,
        index_path=target_dir / "MEMORY.md",
        logs_dir=target_dir / "logs",
        topic_paths=topic_paths,
    )
    store.save_index([])

    updated_at = str(profile.get("meta", {}).get("updated_at") or "")
    topic_payloads = _build_topic_payloads(profile, updated_at=updated_at)
    written_topics: list[str] = []

    for topic_name, payload in topic_payloads.items():
        lines = payload["lines"]
        if not lines:
            continue
        metadata = payload["metadata"]
        summary = payload["summary"]
        store.write_topic(topic_name, lines, metadata)
        store.update_index_entry(topic_name, summary)
        written_topics.append(topic_name)

    return ProfileMigrationResult(
        written_topics=written_topics,
        index_entries=len(store.load_index()),
    )


def _load_profile(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Profile payload must be a JSON object")
    return payload


def _build_topic_payloads(profile: dict[str, Any], *, updated_at: str) -> dict[str, dict[str, object]]:
    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    preferences = profile.get("preferences", {}) if isinstance(profile.get("preferences"), dict) else {}
    workflow = profile.get("workflow", {}) if isinstance(profile.get("workflow"), dict) else {}
    tool_preferences = profile.get("tool_preferences", {}) if isinstance(profile.get("tool_preferences"), dict) else {}
    meta = profile.get("meta", {}) if isinstance(profile.get("meta"), dict) else {}
    counters = meta.get("workflow_counters", {}) if isinstance(meta.get("workflow_counters"), dict) else {}
    recipient_counter = counters.get("common_recipients", {}) if isinstance(counters.get("common_recipients"), dict) else {}

    topic_payloads: dict[str, dict[str, object]] = {}

    identity_lines: list[str] = []
    name = str(identity.get("name") or "").strip()
    if name:
        identity_lines.append(f"- Name: {name}")
    emails = [str(email).strip().lower() for email in identity.get("emails", []) if str(email).strip()]
    if emails:
        identity_lines.append(f"- Primary email: {emails[0]}")
        for email in emails[1:]:
            identity_lines.append(f"- Known email: {email}")
    timezone = str(identity.get("timezone") or "").strip()
    if timezone:
        identity_lines.append(f"- Timezone: {timezone}")
    topic_payloads["identity"] = {
        "lines": identity_lines,
        "metadata": {
            "topic": "identity",
            "title": "Identity",
            "confidence": "confirmed",
            "updated_at": updated_at,
            "version": str(meta.get("version", 1)),
            "source_session_id": "profile_migration",
        },
        "summary": "Name, email, timezone, preferred address",
    }

    preference_lines: list[str] = []
    response_style = str(preferences.get("response_style") or "").strip()
    if response_style:
        preference_lines.append(f"- Response style: {response_style}")
    humor_level = str(preferences.get("humor_level") or "").strip()
    if humor_level:
        preference_lines.append(f"- Humor level: {humor_level}")
    email_tone = str(preferences.get("email_tone") or "").strip()
    if email_tone:
        preference_lines.append(f"- Email tone: {email_tone}")
    topic_payloads["preferences"] = {
        "lines": preference_lines,
        "metadata": {
            "topic": "preference",
            "title": "Preferences",
            "confidence": "confirmed",
            "updated_at": updated_at,
            "version": str(meta.get("version", 1)),
            "source_session_id": "profile_migration",
        },
        "summary": "Tone, response style, and delivery defaults",
    }

    workflow_lines: list[str] = []
    if workflow.get("prefers_draft_before_send") is not None:
        workflow_lines.append(
            f"- Prefers draft before send: {str(bool(workflow.get('prefers_draft_before_send'))).lower()}"
        )
    email_interactions = int(workflow.get("email_interactions", 0) or 0)
    if email_interactions:
        workflow_lines.append(f"- Email interactions recorded: {email_interactions}")
    primary_llm = str(tool_preferences.get("primary_llm") or "").strip()
    if primary_llm:
        workflow_lines.append(f"- Preferred primary model: {primary_llm}")
    topic_payloads["workflow"] = {
        "lines": workflow_lines,
        "metadata": {
            "topic": "workflow",
            "title": "Workflow",
            "confidence": "confirmed",
            "updated_at": updated_at,
            "version": str(meta.get("version", 1)),
            "source_session_id": "profile_migration",
        },
        "summary": "Recurring habits and operational defaults",
    }

    contact_lines: list[str] = []
    common_recipients = [
        str(recipient).strip().lower()
        for recipient in workflow.get("common_recipients", [])
        if str(recipient).strip()
    ]
    counter_items = sorted(
        (
            (str(recipient).strip().lower(), int(count))
            for recipient, count in recipient_counter.items()
            if str(recipient).strip()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    emitted_recipients: set[str] = set()
    for recipient in common_recipients:
        if recipient in emitted_recipients:
            continue
        emitted_recipients.add(recipient)
        count = next((item_count for item_recipient, item_count in counter_items if item_recipient == recipient), None)
        if count:
            contact_lines.append(f"- Frequent recipient: {recipient} (count: {count})")
        else:
            contact_lines.append(f"- Frequent recipient: {recipient}")
    for recipient, count in counter_items:
        if recipient in emitted_recipients:
            continue
        emitted_recipients.add(recipient)
        contact_lines.append(f"- Frequent recipient: {recipient} (count: {count})")
    topic_payloads["contacts"] = {
        "lines": contact_lines,
        "metadata": {
            "topic": "contact",
            "title": "Contacts",
            "confidence": "confirmed",
            "updated_at": updated_at,
            "version": str(meta.get("version", 1)),
            "source_session_id": "profile_migration",
        },
        "summary": "Frequent recipients and confirmed aliases",
    }

    topic_payloads["projects"] = {
        "lines": [],
        "metadata": {
            "topic": "project",
            "title": "Projects",
            "confidence": "confirmed",
            "updated_at": updated_at,
            "version": str(meta.get("version", 1)),
            "source_session_id": "profile_migration",
        },
        "summary": "Project context and recurring work references",
    }

    return topic_payloads


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate profile.json into Markdown-backed personal memory files.")
    parser.add_argument("--source", type=Path, default=MEMORY_PROFILE_FILE, help="Path to source profile.json")
    parser.add_argument("--target-dir", type=Path, default=PERSONAL_MEMORY_DIR, help="Target personal memory directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Markdown personal memory files in the target directory")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = migrate_profile_to_markdown(
        profile_path=args.source,
        target_dir=args.target_dir,
        force=args.force,
    )
    print(f"Migrated topics: {', '.join(result.written_topics) if result.written_topics else 'none'}")
    print(f"Index entries: {result.index_entries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
