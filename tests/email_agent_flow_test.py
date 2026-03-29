import unittest
import uuid
import shutil
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.exceptions import ModelRetry

from core.email_flow import (
    build_send_request,
    extract_deterministic_email_details,
    format_missing_email_prompt,
    merge_email_details,
    missing_email_fields,
    normalize_spoken_email_text,
    parse_email_extraction_response,
    send_email_now,
    validate_recipients,
    validate_send_email_args,
)
from core.session_store import SessionStore


class DummyEmailTool:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def send_email(self, receiver: str, subject: str, body: str, content_type: str = "plain") -> str:
        self.calls.append(
            {
                "receiver": receiver,
                "subject": subject,
                "body": body,
                "content_type": content_type,
            }
        )
        return self.result


class EmailFlowTests(unittest.TestCase):
    def test_spoken_email_normalization_and_extraction(self) -> None:
        text = (
            "Send an email to shriyashbeohar1 at the rate gmail dot com "
            "subject is hello body is world"
        )
        normalized = normalize_spoken_email_text(text)
        details = extract_deterministic_email_details(text)

        self.assertIn("shriyashbeohar1@gmail.com", normalized)
        self.assertEqual(details["recipients"], ["shriyashbeohar1@gmail.com"])
        self.assertEqual(details["subject"], "hello")
        self.assertEqual(details["content"], "world")
        self.assertTrue(details["send_intent"])

    def test_partial_turns_merge_cleanly(self) -> None:
        base = {
            "recipients": ["old@example.com"],
            "subject": "",
            "content": "",
            "send_intent": False,
        }
        first = {
            "recipients": ["new@example.com"],
            "subject": "hello",
            "content": "",
            "send_intent": False,
        }
        second = {
            "recipients": [],
            "subject": "",
            "content": "how are you doing?",
            "send_intent": True,
        }

        merged = merge_email_details(base, first)
        merged = merge_email_details(merged, second)

        self.assertEqual(merged["recipients"], ["new@example.com"])
        self.assertEqual(merged["subject"], "hello")
        self.assertEqual(merged["content"], "how are you doing?")
        self.assertTrue(merged["send_intent"])

    def test_plain_json_extraction_response_parses_cleanly(self) -> None:
        parsed = parse_email_extraction_response(
            '{"recipients":["user@example.com"],"subject":"hello","content":"world","send_intent":true}'
        )
        self.assertEqual(parsed.recipients, ["user@example.com"])
        self.assertEqual(parsed.subject, "hello")
        self.assertEqual(parsed.content, "world")
        self.assertTrue(parsed.send_intent)

    def test_fenced_json_extraction_response_parses_cleanly(self) -> None:
        parsed = parse_email_extraction_response(
            '```json\n{"recipients":["user@example.com"],"subject":"hello","content":"world","send_intent":false}\n```'
        )
        self.assertEqual(parsed.recipients, ["user@example.com"])
        self.assertEqual(parsed.subject, "hello")
        self.assertEqual(parsed.content, "world")
        self.assertFalse(parsed.send_intent)

    def test_missing_field_prompt_only_requests_gaps(self) -> None:
        details = {
            "recipients": ["user@example.com"],
            "subject": "hello",
            "content": "",
        }
        missing = missing_email_fields(details)
        prompt = format_missing_email_prompt(missing, details)

        self.assertEqual(missing, ["content"])
        self.assertIn("To: user@example.com", prompt)
        self.assertIn("Subject: hello", prompt)
        self.assertIn("missing email body/message", prompt)

    def test_recipient_validation_splits_valid_and_invalid(self) -> None:
        valid, invalid = validate_recipients(["good@example.com", "bad-email", "also@good.com"])
        self.assertEqual(valid, ["good@example.com", "also@good.com"])
        self.assertEqual(invalid, ["bad-email"])

    def test_send_validator_rejects_missing_or_invalid_fields(self) -> None:
        with self.assertRaises(ModelRetry):
            validate_send_email_args([], "hello", "world")
        with self.assertRaises(ModelRetry):
            validate_send_email_args(["bad-email"], "hello", "world")
        with self.assertRaises(ModelRetry):
            validate_send_email_args(["user@example.com"], "", "world")
        with self.assertRaises(ModelRetry):
            validate_send_email_args(["user@example.com"], "hello", "")

    def test_send_request_payload_is_exact(self) -> None:
        request = build_send_request(
            {
                "recipients": ["user@example.com"],
                "subject": "hello",
                "content": "world",
            }
        )
        self.assertIn('"recipients": ["user@example.com"]', request)
        self.assertIn('"subject": "hello"', request)
        self.assertIn('"content": "world"', request)

    def test_send_email_now_uses_tool_and_wraps_success(self) -> None:
        dummy = DummyEmailTool(" Email sent successfully to user@example.com\nSubject: hello\nStatus: ok")
        with patch("core.email_flow.create_email_tool_from_env", return_value=dummy):
            result = send_email_now(
                {
                    "recipients": ["user@example.com"],
                    "subject": "hello",
                    "content": "world",
                }
            )

        self.assertIn("Email sent successfully!", result)
        self.assertEqual(len(dummy.calls), 1)
        self.assertEqual(dummy.calls[0]["receiver"], "user@example.com")
        self.assertEqual(dummy.calls[0]["subject"], "hello")
        self.assertIn("world", dummy.calls[0]["body"])

    def test_session_store_persists_pending_email(self) -> None:
        base = Path("tests") / "_tmp" / f"email_session_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            manifest = base / "session.json"
            messages = base / "messages.json"
            archive = base / "archive"

            store = SessionStore(manifest_path=manifest, messages_path=messages, archive_dir=archive)
            store.start_or_restore()
            store.set_pending_email(
                recipients=["user@example.com"],
                subject="hello",
                content="world",
            )

            restored = SessionStore(manifest_path=manifest, messages_path=messages, archive_dir=archive)
            restored.start_or_restore(mode="resume_if_active")

            self.assertEqual(
                restored.get_pending_email(),
                {
                    "recipients": ["user@example.com"],
                    "subject": "hello",
                    "content": "world",
                },
            )
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
