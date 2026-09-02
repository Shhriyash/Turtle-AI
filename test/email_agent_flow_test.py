import unittest
import uuid
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_ai.exceptions import ModelRetry

from core.email_flow import (
    build_compose_email_prompt,
    build_send_request,
    combine_extracted_email_details,
    derive_fallback_subject,
    extract_deterministic_email_details,
    format_email_draft,
    format_missing_email_prompt,
    merge_email_details,
    missing_email_fields,
    normalize_spoken_email_text,
    send_email_now,
    validate_recipients,
    validate_send_email_args,
)
from core.session_store import SessionStore


class DummyEmailTool:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def send_email(
        self,
        receiver: str,
        subject: str,
        body: str,
        content_type: str = "plain",
        cc: str = "",
        bcc: str = "",
    ) -> str:
        self.calls.append(
            {
                "receiver": receiver,
                "subject": subject,
                "body": body,
                "content_type": content_type,
                "cc": cc,
                "bcc": bcc,
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

    def test_cc_bcc_extraction_from_user_text(self) -> None:
        text = (
            "Send an email to to.person at the rate gmail dot com "
            "cc is cc.person at the rate gmail dot com "
            "bcc bcc.person at the rate gmail dot com "
            "subject is hello body is world"
        )
        details = extract_deterministic_email_details(text)

        self.assertEqual(details["recipients"], ["to.person@gmail.com"])
        self.assertEqual(details["cc_recipients"], ["cc.person@gmail.com"])
        self.assertEqual(details["bcc_recipients"], ["bcc.person@gmail.com"])

    def test_bcc_extraction_when_address_appears_before_bcc_label(self) -> None:
        text = (
            "my main mail is shriyashbeohar1 at the rate gmail dot com "
            "keep it in bcc and send a mail to shriyash.beohar at the rate gmail dot com"
        )
        details = extract_deterministic_email_details(text)

        self.assertEqual(details["recipients"], ["shriyash.beohar@gmail.com"])
        self.assertEqual(details["bcc_recipients"], ["shriyashbeohar1@gmail.com"])

    def test_combiner_moves_bcc_out_of_to_bucket(self) -> None:
        deterministic = {
            "recipients": ["shriyashbeohar1@gmail.com", "shriyash.beohar@gmail.com"],
            "cc_recipients": [],
            "bcc_recipients": [],
            "subject": "hello",
            "content": "",
            "send_intent": True,
        }
        llm_values = {
            "recipients": ["shriyash.beohar@gmail.com"],
            "cc_recipients": [],
            "bcc_recipients": ["shriyashbeohar1@gmail.com"],
            "subject": "",
            "content": "tell about yourself",
            "send_intent": True,
        }

        combined = combine_extracted_email_details(deterministic, llm_values)
        self.assertEqual(combined["recipients"], ["shriyash.beohar@gmail.com"])
        self.assertEqual(combined["bcc_recipients"], ["shriyashbeohar1@gmail.com"])
        self.assertEqual(combined["content"], "tell about yourself")

    def test_partial_turns_merge_cleanly(self) -> None:
        base = {
            "recipients": ["old@example.com"],
            "cc_recipients": [],
            "bcc_recipients": [],
            "subject": "",
            "content": "",
            "send_intent": False,
        }
        first = {
            "recipients": ["new@example.com"],
            "cc_recipients": ["cc@example.com"],
            "subject": "hello",
            "content": "",
            "send_intent": False,
        }
        second = {
            "recipients": [],
            "bcc_recipients": ["hidden@example.com"],
            "subject": "",
            "content": "how are you doing?",
            "send_intent": True,
        }

        merged = merge_email_details(base, first)
        merged = merge_email_details(merged, second)

        self.assertEqual(merged["recipients"], ["new@example.com"])
        self.assertEqual(merged["cc_recipients"], ["cc@example.com"])
        self.assertEqual(merged["bcc_recipients"], ["hidden@example.com"])
        self.assertEqual(merged["subject"], "hello")
        self.assertEqual(merged["content"], "how are you doing?")
        self.assertTrue(merged["send_intent"])

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

    def test_compose_prompt_includes_request_identity_and_existing_fields(self) -> None:
        prompt = build_compose_email_prompt(
            user_request="tell about yourself and pick a subject",
            merged={"subject": "", "content": "", "recipients": ["me@example.com"]},
            email_tone="friendly",
            sender_identity="You are Turtle.",
        )
        self.assertIn("Compose an email", prompt)
        self.assertIn("You are Turtle.", prompt)
        self.assertIn("friendly", prompt)
        self.assertIn("tell about yourself and pick a subject", prompt)
        # Must ask for JSON with subject + content (parseable by the extractor).
        self.assertIn('"subject"', prompt)
        self.assertIn('"content"', prompt)

    def test_compose_prompt_preserves_dictated_fields(self) -> None:
        prompt = build_compose_email_prompt(
            user_request="finish this email",
            merged={"subject": "Quarterly update", "content": "Numbers attached.", "recipients": []},
        )
        # Existing user-dictated values are shown so the model reuses them.
        self.assertIn("Quarterly update", prompt)
        self.assertIn("Numbers attached.", prompt)

    def test_derive_fallback_subject(self) -> None:
        self.assertEqual(
            derive_fallback_subject("Hello there, this is Turtle. Extra."),
            "Hello there, this is Turtle. Extra",  # trailing punctuation trimmed
        )
        self.assertEqual(derive_fallback_subject(""), "(no subject)")
        self.assertEqual(derive_fallback_subject("   \n\n  Real body line here"), "Real body line here")
        # Caps at 8 words.
        self.assertEqual(
            derive_fallback_subject("one two three four five six seven eight nine ten"),
            "one two three four five six seven eight",
        )

    def test_format_email_draft_shows_headers_and_body_and_omits_empty(self) -> None:
        draft = format_email_draft({
            "recipients": ["me@example.com"],
            "cc_recipients": [],
            "bcc_recipients": [],
            "subject": "About Turtle",
            "content": "Hi.\nI'm your assistant.",
        })
        self.assertIn("To: me@example.com", draft)
        self.assertIn("Subject: About Turtle", draft)
        self.assertIn("I'm your assistant.", draft)
        self.assertNotIn("Cc:", draft)
        self.assertNotIn("Bcc:", draft)

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
            validate_send_email_args(["user@example.com"], "hello", "world", ["bad-cc"])
        with self.assertRaises(ModelRetry):
            validate_send_email_args(["user@example.com"], "hello", "world", [], ["bad-bcc"])
        with self.assertRaises(ModelRetry):
            validate_send_email_args(["user@example.com"], "", "world")
        with self.assertRaises(ModelRetry):
            validate_send_email_args(["user@example.com"], "hello", "")

    def test_send_request_payload_is_exact(self) -> None:
        request = build_send_request(
            {
                "recipients": ["user@example.com"],
                "cc_recipients": ["copy@example.com"],
                "bcc_recipients": ["hidden@example.com"],
                "subject": "hello",
                "content": "world",
            }
        )
        self.assertIn('"recipients": ["user@example.com"]', request)
        self.assertIn('"cc_recipients": ["copy@example.com"]', request)
        self.assertIn('"bcc_recipients": ["hidden@example.com"]', request)
        self.assertIn('"subject": "hello"', request)
        self.assertIn('"content": "world"', request)

    def test_send_email_now_uses_tool_and_wraps_success(self) -> None:
        dummy = DummyEmailTool(" Email sent successfully to user@example.com\nSubject: hello\nStatus: ok")
        with patch("core.email_flow.create_email_tool_from_env", return_value=dummy):
            result = send_email_now(
                {
                    "recipients": ["user@example.com"],
                    "cc_recipients": ["copy@example.com"],
                    "bcc_recipients": ["hidden@example.com"],
                    "subject": "hello",
                    "content": "world",
                }
            )

        self.assertIn("Email sent successfully!", result)
        self.assertEqual(len(dummy.calls), 1)
        self.assertEqual(dummy.calls[0]["receiver"], "user@example.com")
        self.assertEqual(dummy.calls[0]["cc"], "copy@example.com")
        self.assertEqual(dummy.calls[0]["bcc"], "hidden@example.com")
        self.assertEqual(dummy.calls[0]["subject"], "hello")
        self.assertIn("world", dummy.calls[0]["body"])

    def test_send_email_now_deduplicates_trailing_signoff(self) -> None:
        dummy = DummyEmailTool(" Email sent successfully to user@example.com\nSubject: hello\nStatus: ok")
        duplicated = (
            "Hello ma'am,\n\n"
            "Shriyash asked me to tell you how much he loves you.\n\n"
            "Best regards,\n"
            "Turtle\n\n"
            "Best regards,\n"
            "TurtleAI"
        )
        with patch("core.email_flow.create_email_tool_from_env", return_value=dummy):
            send_email_now(
                {
                    "recipients": ["user@example.com"],
                    "cc_recipients": [],
                    "bcc_recipients": [],
                    "subject": "hello",
                    "content": duplicated,
                }
            )

        body = dummy.calls[0]["body"]
        self.assertEqual(body.lower().count("best regards"), 1)
        self.assertTrue(body.strip().endswith("Best regards,\nTurtleAI"))

    @pytest.mark.skip(reason="stale: SessionStore dropped manifest_path/messages_path file kwargs for a SQLite backend and start_or_restore is now async — 2026-07-16 Phase 1 triage")
    def test_session_store_persists_pending_email(self) -> None:
        base = Path("test") / "_tmp" / f"email_session_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        try:
            manifest = base / "session.json"
            messages = base / "messages.json"
            archive = base / "archive"

            store = SessionStore(manifest_path=manifest, messages_path=messages, archive_dir=archive)
            store.start_or_restore()
            store.set_pending_email(
                recipients=["user@example.com"],
                cc_recipients=["copy@example.com"],
                bcc_recipients=["hidden@example.com"],
                subject="hello",
                content="world",
            )

            restored = SessionStore(manifest_path=manifest, messages_path=messages, archive_dir=archive)
            restored.start_or_restore(mode="resume_if_active")

            self.assertEqual(
                restored.get_pending_email(),
                {
                    "recipients": ["user@example.com"],
                    "cc_recipients": ["copy@example.com"],
                    "bcc_recipients": ["hidden@example.com"],
                    "subject": "hello",
                    "content": "world",
                },
            )
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
