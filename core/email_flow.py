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
    subject: str = ""
    content: str = ""
    send_intent: bool = False


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

    recipients = payload.get("recipients", [])
    if isinstance(recipients, str):
        recipients = [recipients]
    elif not isinstance(recipients, list):
        recipients = []

    return EmailExtractionOutput(
        recipients=[str(item).strip() for item in recipients if str(item).strip()],
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


def extract_recipients(text: str) -> list[str]:
    normalized = normalize_spoken_email_text(text)
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return dedupe_strings(re.findall(pattern, normalized))


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
    return {
        "recipients": extract_recipients(normalized),
        "subject": subject,
        "content": content,
        "send_intent": send_intent,
    }


def sanitize_email_details(details: dict[str, Any]) -> dict[str, Any]:
    recipients = [addr.strip() for addr in details.get("recipients", []) if str(addr).strip()]
    subject = str(details.get("subject", "")).strip()
    content = str(details.get("content", "")).strip()
    send_intent = bool(details.get("send_intent", False))
    return {
        "recipients": dedupe_strings(recipients),
        "subject": subject,
        "content": content,
        "send_intent": send_intent,
    }


def merge_email_details(base: dict[str, Any], new_values: dict[str, Any]) -> dict[str, Any]:
    merged = sanitize_email_details(base)
    incoming = sanitize_email_details(new_values)
    if incoming["recipients"]:
        merged["recipients"] = incoming["recipients"]
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
    if details.get("subject"):
        captured_parts.append(f"Subject: {details['subject']}")
    if details.get("content"):
        captured_parts.append(f"Body: {details['content']}")
    captured_text = "\n".join(captured_parts)
    if captured_text:
        return f"I have these details already:\n{captured_text}\n\nPlease provide the missing {missing_text}."
    return f"Please provide the missing {missing_text} so I can send the email."


def send_email_now(details: dict[str, Any]) -> str:
    email_tool = create_email_tool_from_env()
    if not email_tool:
        return (
            "Email configuration missing. Please set up TURTLE_EMAIL_NAME, "
            "TURTLE_EMAIL_ADDRESS, and TURTLE_EMAIL_PASSKEY environment variables."
        )
    enhanced_content = f"\n\n{details['content']}\n\nBest regards,\nTurtleAI"
    recipients = ",".join(details["recipients"])
    result = email_tool.send_email(
        receiver=recipients,
        subject=details["subject"],
        body=enhanced_content,
        content_type="plain",
    )
    if result.startswith("error:"):
        return f"Failed to send email: {result}"
    return (
        "Email sent successfully!\n\n"
        f"To: {', '.join(details['recipients'])}\n"
        f"Subject: {details['subject']}\n\n"
        f"{result}"
    )


def validate_send_email_args(
    recipients: list[str],
    subject: str,
    content: str,
) -> None:
    cleaned_recipients = [addr.strip() for addr in recipients if addr.strip()]
    if not cleaned_recipients:
        raise ModelRetry("Recipient email address is missing. Ask for the recipient again.")
    invalid_recipients = [addr for addr in cleaned_recipients if not EmailRequest.is_valid_email(addr)]
    if invalid_recipients:
        invalid_text = ", ".join(invalid_recipients)
        raise ModelRetry(
            f"Invalid recipient email format: {invalid_text}. "
            "Use only valid email addresses."
        )
    if not subject.strip():
        raise ModelRetry("Subject is missing. Ask for the subject line.")
    if not content.strip():
        raise ModelRetry("Email body is missing. Ask for the body/content.")


def build_send_request(details: dict[str, Any]) -> str:
    payload = {
        "recipients": details["recipients"],
        "subject": details["subject"],
        "content": details["content"],
    }
    return (
        "Send the email using the tool with these exact values.\n"
        "Do not invent or alter any field.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
