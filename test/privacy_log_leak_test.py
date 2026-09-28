"""
test/privacy_log_leak_test.py
------------------------------
Four stdout print() calls leaked user PII into logs that Vercel retains and
an operator can view: a plaintext email address on identity rebind
(core/storage/cloud/identity_store.py), and the outgoing message text/body
on all three channel adapters when credentials are unconfigured
(apps/channels/{imessage,whatsapp,slack}.py).

These tests drive the real functions (not source-grepping) and assert the
plaintext PII does NOT appear in captured stdout. Run against the
pre-fix code, each of these fails by actually observing the leaked value in
capsys output (see the WP report for quoted failures).
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import pytest


# --- 1. identity_store._rebind_from_markers leaks the email --------------

class IdentityRebindEmailLeakTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Reuse the fake asyncpg pool already built for this module's own
        # unit tests rather than re-implementing one.
        from test.identity_store_cloud_test import _FakeAsyncPool
        from core.storage.cloud.identity_store import PostgresIdentityManager

        self.pool = _FakeAsyncPool()
        patcher = patch(
            "core.storage.cloud.identity_store.get_pg_pool",
            new_callable=AsyncMock,
            return_value=self.pool,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.manager = PostgresIdentityManager()

    async def test_rebind_does_not_print_plaintext_email(self) -> None:
        email = "alice.secret@example.com"
        # Seed a verified marker under some other user_id so resolve_user's
        # unknown-channel path falls into _rebind_from_markers.
        self.pool.seed_marker("usr_existing", email, verified=True)

        with _capture_stdout() as buf:
            user_id = await self.manager.resolve_user("web_email", email)

        self.assertEqual(user_id, "usr_existing")
        output = buf.getvalue()
        assert email not in output, f"plaintext email leaked into stdout: {output!r}"


# --- helpers ---------------------------------------------------------------

import contextlib
import io


@contextlib.contextmanager
def _capture_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# --- 2/3/4. channel adapters leak outgoing message text/body -------------

def test_imessage_send_does_not_print_message_text(capsys):
    from apps.channels.imessage import _send_imessage_reply

    secret_text = "my social security number is 123-45-6789"
    with patch("apps.channels.imessage.settings") as s:
        s.sendblue_api_key = None
        s.sendblue_api_secret = None
        asyncio.run(_send_imessage_reply("+15551234567", secret_text))

    out = capsys.readouterr().out
    assert secret_text not in out, f"message text leaked into stdout: {out!r}"


def test_whatsapp_send_does_not_print_message_body(capsys):
    from apps.channels.whatsapp import _send_whatsapp_reply

    secret_body = "here is my password: hunter2-super-secret"
    with patch("apps.channels.whatsapp.settings") as s:
        s.twilio_account_sid = None
        s.twilio_auth_token = None
        s.twilio_whatsapp_number = None
        asyncio.run(_send_whatsapp_reply("+15551234567", secret_body))

    out = capsys.readouterr().out
    assert secret_body not in out, f"message body leaked into stdout: {out!r}"


def test_slack_send_does_not_print_message_text(capsys):
    from apps.channels.slack import _post_slack_message

    secret_text = "the API key is sk-verysecretvalue-do-not-log"
    with patch("apps.channels.slack.settings") as s:
        s.slack_bot_token = None
        asyncio.run(_post_slack_message("C123456", secret_text))

    out = capsys.readouterr().out
    assert secret_text not in out, f"message text leaked into stdout: {out!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
