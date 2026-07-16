from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.exceptions import ModelRetry

from tools.email_tools.config import create_email_tool_from_env
from tools.email_tools.models import EmailRequest


class EmailExtractionOutput(BaseModel):
    """Structured extraction output from email specialist."""

    recipients: list[str] = Field(default_factory=list)
    cc_recipients: list[str] = Field(default_factory=list)
    bcc_recipients: list[str] = Field(default_factory=list)
    subject: str = ""
    content: str = ""
    send_intent: bool = False


_SIGNOFF_PREFIXES = (
    "best regards",
    "kind regards",
    "regards",
    "sincerely",
    "warm regards",
    "thanks and regards",
    "thanks",
    "thank you",
    "cheers",
    "with love",
)


def _coerce_email_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return extract_recipients(value)
    if isinstance(value, list):
        extracted: list[str] = []
        for item in value:
            extracted.extend(extract_recipients(str(item)))
        return dedupe_strings(extracted)
    return []


def parse_email_extraction_response(response_text: str) -> EmailExtractionOutput:
    """Parse a plain-text LLM response into EmailExtractionOutput.

    The extractor prompt asks for JSON only. This parser still tolerates:
    - fenced ```json blocks
    - leading/trailing prose around a JSON object
    - partial/malformed output by falling back to empty fields
    """
    text = response_text.strip()
    if not text:
        return EmailExtractionOutput()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return EmailExtractionOutput()

    if not isinstance(payload, dict):
        return EmailExtractionOutput()

    recipients = _coerce_email_list(payload.get("recipients", []))
    cc_recipients = _coerce_email_list(payload.get("cc_recipients", payload.get("cc", [])))
    bcc_recipients = _coerce_email_list(payload.get("bcc_recipients", payload.get("bcc", [])))

    return EmailExtractionOutput(
        recipients=recipients,
        cc_recipients=cc_recipients,
        bcc_recipients=bcc_recipients,
        subject=str(payload.get("subject", "")).strip(),
        content=str(payload.get("content", "")).strip(),
        send_intent=bool(payload.get("send_intent", False)),
    )


def normalize_spoken_email_text(text: str) -> str:
    normalized = text.strip()
    replacements = [
        (r"\bat the rate\b", "@"),
        (r"\battherate\b", "@"),
        (r"\bat rate\b", "@"),
        (r"\(at\)", "@"),
        (r"\bdot\b", "."),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*@\s*", "@", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    return normalized


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(value)
    return output


def normalize_recipient_buckets(details: dict[str, Any]) -> dict[str, Any]:
    recipients = dedupe_strings([str(addr).strip() for addr in details.get("recipients", []) if str(addr).strip()])
    cc_recipients = dedupe_strings([str(addr).strip() for addr in details.get("cc_recipients", []) if str(addr).strip()])
    bcc_recipients = dedupe_strings([str(addr).strip() for addr in details.get("bcc_recipients", []) if str(addr).strip()])

    bcc_lower = {addr.lower() for addr in bcc_recipients}
    cc_recipients = [addr for addr in cc_recipients if addr.lower() not in bcc_lower]
    cc_lower = {addr.lower() for addr in cc_recipients}
    recipients = [addr for addr in recipients if addr.lower() not in cc_lower and addr.lower() not in bcc_lower]

    normalized = dict(details)
    normalized["recipients"] = recipients
    normalized["cc_recipients"] = cc_recipients
    normalized["bcc_recipients"] = bcc_recipients
    return normalized


def extract_recipients(text: str) -> list[str]:
    normalized = normalize_spoken_email_text(text)
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return dedupe_strings(re.findall(pattern, normalized))


def extract_labeled_recipients(text: str, label: str) -> list[str]:
    normalized = normalize_spoken_email_text(text)
    common_stop_pattern = (
        r"(?:subject(?: is)?|body(?: is)?|content(?: is)?|message(?: is)?|"
        r"send(?:\s+a|\s+an)?\s+(?:mail|email)\s+to|send\s+to|mail\s+to|email\s+to)"
    )
    if label == "cc":
        marker_pattern = r"(?:cc|copy to|carbon copy(?: to)?|in cc|add cc)"
        stop_pattern = rf"(?:bcc|{common_stop_pattern})"
    elif label == "bcc":
        marker_pattern = r"(?:bcc|blind carbon copy(?: to)?|in bcc|add bcc)"
        stop_pattern = rf"(?:cc|{common_stop_pattern})"
    else:
        return []

    pattern = re.compile(
        rf"\b{marker_pattern}\b\s*(?::|is|to|as)?\s*(.*?)(?=\b{stop_pattern}\b|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    emails: list[str] = []
    for match in pattern.finditer(normalized):
        segment = match.group(1).strip()
        if not segment:
            continue
        emails.extend(extract_recipients(segment))
    return dedupe_strings(emails)


def _extract_post_labeled_recipients(text: str, label: str, emails: list[str]) -> list[str]:
    collected: list[str] = []
    for email in emails:
        escaped = re.escape(email)
        direct_pattern = re.compile(
            rf"{escaped}.{{0,60}}\b(?:in|as)\s*(?:the\s+)?{label}\b",
            flags=re.IGNORECASE | re.DOTALL,
        )
        action_pattern = re.compile(
            rf"{escaped}.{{0,120}}\b(?:keep|put|add|mark|set)\b.{{0,40}}\b(?:in|as)?\s*(?:the\s+)?{label}\b",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if direct_pattern.search(text) or action_pattern.search(text):
            collected.append(email)
    return dedupe_strings(collected)


def extract_segment(text: str, markers: list[str], stop_tokens: list[str]) -> str:
    lowered = text.lower()
    for marker in markers:
        idx = lowered.find(marker)
        if idx < 0:
            continue
        start = idx + len(marker)
        end = len(text)
        for stop in stop_tokens:
            stop_idx = lowered.find(stop, start)
            if stop_idx >= 0:
                end = min(end, stop_idx)
        segment = text[start:end].strip(" .,:;\"'")
        if segment:
            return segment
    return ""


def extract_deterministic_email_details(text: str) -> dict[str, Any]:
    normalized = normalize_spoken_email_text(text)
    all_emails = extract_recipients(normalized)
    cc_recipients = extract_labeled_recipients(normalized, "cc")
    bcc_recipients = extract_labeled_recipients(normalized, "bcc")
    cc_recipients = dedupe_strings(cc_recipients + _extract_post_labeled_recipients(normalized, "cc", all_emails))
    bcc_recipients = dedupe_strings(bcc_recipients + _extract_post_labeled_recipients(normalized, "bcc", all_emails))
    to_recipients = [
        email
        for email in all_emails
        if email not in cc_recipients and email not in bcc_recipients
    ]
    subject = extract_segment(
        normalized,
        markers=["subject is", "subject:", "with subject", "title is"],
        stop_tokens=[" and body", " and content", " and message", " body is", " body:", "\n"],
    )
    content = extract_segment(
        normalized,
        markers=["body is", "content is", "message is", "body:", "content:", "saying"],
        stop_tokens=[" and subject", "\n"],
    )
    send_intent = bool(
        re.search(
            r"\b(send|mail|email it|fire off|shoot over|dispatch)\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    return normalize_recipient_buckets(
        {
        "recipients": to_recipients,
        "cc_recipients": cc_recipients,
        "bcc_recipients": bcc_recipients,
        "subject": subject,
        "content": content,
        "send_intent": send_intent,
        }
    )


def sanitize_email_details(details: dict[str, Any]) -> dict[str, Any]:
    recipients = [addr.strip() for addr in details.get("recipients", []) if str(addr).strip()]
    cc_recipients = [addr.strip() for addr in details.get("cc_recipients", []) if str(addr).strip()]
    bcc_recipients = [addr.strip() for addr in details.get("bcc_recipients", []) if str(addr).strip()]
    subject = str(details.get("subject", "")).strip()
    content = str(details.get("content", "")).strip()
    send_intent = bool(details.get("send_intent", False))
    return normalize_recipient_buckets(
        {
        "recipients": dedupe_strings(recipients),
        "cc_recipients": dedupe_strings(cc_recipients),
        "bcc_recipients": dedupe_strings(bcc_recipients),
        "subject": subject,
        "content": content,
        "send_intent": send_intent,
        }
    )


def combine_extracted_email_details(deterministic: dict[str, Any], llm_values: dict[str, Any]) -> dict[str, Any]:
    deterministic_details = sanitize_email_details(deterministic)
    llm_details = sanitize_email_details(llm_values)
    return sanitize_email_details(
        {
            "recipients": deterministic_details["recipients"] + llm_details["recipients"],
            "cc_recipients": deterministic_details["cc_recipients"] + llm_details["cc_recipients"],
            "bcc_recipients": deterministic_details["bcc_recipients"] + llm_details["bcc_recipients"],
            "subject": deterministic_details["subject"] or llm_details["subject"],
            "content": deterministic_details["content"] or llm_details["content"],
            "send_intent": bool(deterministic_details["send_intent"] or llm_details["send_intent"]),
        }
    )


def merge_email_details(base: dict[str, Any], new_values: dict[str, Any]) -> dict[str, Any]:
    merged = sanitize_email_details(base)
    incoming = sanitize_email_details(new_values)
    if incoming["recipients"]:
        merged["recipients"] = incoming["recipients"]
    if incoming["cc_recipients"]:
        merged["cc_recipients"] = incoming["cc_recipients"]
    if incoming["bcc_recipients"]:
        merged["bcc_recipients"] = incoming["bcc_recipients"]
    if incoming["subject"]:
        merged["subject"] = incoming["subject"]
    if incoming["content"]:
        merged["content"] = incoming["content"]
    merged["send_intent"] = bool(merged.get("send_intent") or incoming.get("send_intent"))
    return merged


def validate_recipients(recipients: list[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    invalid: list[str] = []
    for email in recipients:
        if EmailRequest.is_valid_email(email):
            valid.append(email)
        else:
            invalid.append(email)
    return valid, invalid


def missing_email_fields(details: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not details.get("recipients"):
        missing.append("recipients")
    if not details.get("subject"):
        missing.append("subject")
    if not details.get("content"):
        missing.append("content")
    return missing


def format_missing_email_prompt(missing: list[str], details: dict[str, Any]) -> str:
    missing_map = {
        "recipients": "recipient email address",
        "subject": "subject line",
        "content": "email body/message",
    }
    missing_text = ", ".join(missing_map[item] for item in missing)
    captured_parts = []
    if details.get("recipients"):
        captured_parts.append(f"To: {', '.join(details['recipients'])}")
    if details.get("cc_recipients"):
        captured_parts.append(f"Cc: {', '.join(details['cc_recipients'])}")
    if details.get("bcc_recipients"):
        captured_parts.append(f"Bcc: {', '.join(details['bcc_recipients'])}")
    if details.get("subject"):
        captured_parts.append(f"Subject: {details['subject']}")
    if details.get("content"):
        captured_parts.append(f"Body: {details['content']}")
    captured_text = "\n".join(captured_parts)
    if captured_text:
        return f"I have these details already:\n{captured_text}\n\nPlease provide the missing {missing_text}."
    return f"Please provide the missing {missing_text} so I can send the email."


def build_compose_email_prompt(
    *,
    user_request: str,
    merged: dict[str, Any],
    email_tone: str = "",
    sender_identity: str = "",
) -> str:
    """Prompt the email agent to AUTHOR the subject/body the user delegated.

    This is the composition counterpart to the extraction prompt. It is only
    used when the user asked Turtle to write the email (or pick the subject)
    rather than dictating it verbatim. The model owns the wording; we never
    derive it from brittle templates. The model returns the same JSON shape as
    extraction so ``parse_email_extraction_response`` can read it back.
    """
    existing_subject = merged.get("subject", "")
    existing_content = merged.get("content", "")
    tone_line = f"- Match this tone if natural: {email_tone}.\n" if email_tone else ""
    identity_block = f"{sender_identity}\n\n" if sender_identity else ""
    return (
        "Compose an email on the user's behalf from their request below.\n"
        f"{identity_block}"
        'Return ONLY a JSON object with string keys "subject" and "content".\n'
        "Rules:\n"
        "- Write a complete, ready-to-send body that fulfils the request.\n"
        "- Do NOT add a sign-off or signature; one is appended automatically.\n"
        "- Write a concise, specific subject line that fits the body.\n"
        "- Do NOT fabricate facts (names, dates, numbers) you were not given.\n"
        "- If the request gives no basis to write a body, return an empty "
        'string for "content".\n'
        f"{tone_line}"
        "\nAlready provided — reuse verbatim when non-empty, do not rewrite:\n"
        f"- subject: {existing_subject!r}\n"
        f"- content: {existing_content!r}\n\n"
        f"User request:\n{user_request}"
    )


def derive_fallback_subject(content: str) -> str:
    """Last-resort subject when the model returned a body but no subject.

    Keeps a send from blocking on a missing subject. Uses the opening words of
    the body; the composer normally supplies a better one.
    """
    first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    if not first_line:
        return "(no subject)"
    words = first_line.split()
    snippet = " ".join(words[:8]).rstrip(" .,:;!?")
    return snippet or "(no subject)"


def format_email_draft(details: dict[str, Any]) -> str:
    """Render a human-readable draft preview (headers + body) for confirmation."""
    lines = [f"To: {', '.join(details.get('recipients', []))}"]
    if details.get("cc_recipients"):
        lines.append(f"Cc: {', '.join(details['cc_recipients'])}")
    if details.get("bcc_recipients"):
        lines.append(f"Bcc: {', '.join(details['bcc_recipients'])}")
    lines.append(f"Subject: {details.get('subject', '')}")
    lines.append("")
    lines.append(str(details.get("content", "")).strip())
    return "\n".join(lines)


def _looks_like_signoff_line(line: str) -> bool:
    lowered = re.sub(r"[^a-z ]", "", line.lower()).strip()
    return any(lowered.startswith(prefix) for prefix in _SIGNOFF_PREFIXES)


def _strip_trailing_signoff_blocks(content: str) -> str:
    lines = content.strip().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    # Remove one or more trailing sign-off blocks (e.g., duplicated closings).
    while lines:
        signoff_start: int | None = None
        for idx in range(len(lines) - 1, -1, -1):
            if not _looks_like_signoff_line(lines[idx]):
                continue
            tail_non_empty = [line for line in lines[idx + 1:] if line.strip()]
            if len(tail_non_empty) <= 4 and len(lines) - idx <= 8:
                signoff_start = idx
                break
        if signoff_start is None:
            break
        lines = lines[:signoff_start]
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join(lines).strip()


def _compose_email_body(content: str) -> str:
    body_core = _strip_trailing_signoff_blocks(content)
    if body_core:
        return f"{body_core}\n\nBest regards,\nTurtleAI"
    return "Best regards,\nTurtleAI"


def send_email_now(details: dict[str, Any]) -> str:
    email_tool = create_email_tool_from_env()
    if not email_tool:
        return (
            "Email configuration missing. Please set up TURTLE_EMAIL_NAME, "
            "TURTLE_EMAIL_ADDRESS, and TURTLE_EMAIL_PASSKEY environment variables."
        )
    enhanced_content = _compose_email_body(str(details["content"]))
    recipients = ",".join(details["recipients"])
    cc_recipients = ",".join(details.get("cc_recipients", []))
    bcc_recipients = ",".join(details.get("bcc_recipients", []))
    result = email_tool.send_email(
        receiver=recipients,
        subject=details["subject"],
        body=enhanced_content,
        content_type="plain",
        cc=cc_recipients,
        bcc=bcc_recipients,
    )
    if result.startswith("error:"):
        return f"Failed to send email: {result}"
    header_lines = [f"To: {', '.join(details['recipients'])}"]
    if details.get("cc_recipients"):
        header_lines.append(f"Cc: {', '.join(details['cc_recipients'])}")
    if details.get("bcc_recipients"):
        header_lines.append(f"Bcc: {', '.join(details['bcc_recipients'])}")
    header_lines.append(f"Subject: {details['subject']}")
    header_block = "\n".join(header_lines)
    return (
        "Email sent successfully!\n\n"
        f"{header_block}\n\n"
        f"{result}"
    )


def validate_send_email_args(
    recipients: list[str],
    subject: str,
    content: str,
    cc_recipients: list[str] | None = None,
    bcc_recipients: list[str] | None = None,
) -> None:
    cleaned_recipients = [addr.strip() for addr in recipients if addr.strip()]
    cleaned_cc = [addr.strip() for addr in (cc_recipients or []) if addr.strip()]
    cleaned_bcc = [addr.strip() for addr in (bcc_recipients or []) if addr.strip()]
    if not cleaned_recipients:
        raise ModelRetry("Recipient email address is missing. Ask for the recipient again.")
    invalid_recipients = [
        addr
        for addr in cleaned_recipients + cleaned_cc + cleaned_bcc
        if not EmailRequest.is_valid_email(addr)
    ]
    if invalid_recipients:
        invalid_text = ", ".join(invalid_recipients)
        raise ModelRetry(
            f"Invalid email format: {invalid_text}. "
            "Use only valid email addresses."
        )
    if not subject.strip():
        raise ModelRetry("Subject is missing. Ask for the subject line.")
    if not content.strip():
        raise ModelRetry("Email body is missing. Ask for the body/content.")


def build_send_request(details: dict[str, Any]) -> str:
    payload = {
        "recipients": details["recipients"],
        "cc_recipients": details.get("cc_recipients", []),
        "bcc_recipients": details.get("bcc_recipients", []),
        "subject": details["subject"],
        "content": details["content"],
    }
    return (
        "Send the email using the tool with these exact values.\n"
        "Do not invent or alter any field.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
