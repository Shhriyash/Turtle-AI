"""
test/privacy_log_leak_round2_test.py
--------------------------------------
Follow-up sweep to test/privacy_log_leak_test.py. The first pass fixed four
print() sites named by a recon list; a full sweep of the files this WP
actually owns (apps/channels/telegram_webhook.py, apps/channels/discord.py,
apps/channels/twilio_voice.py, core/storage/cloud/identity_store.py) found
four more sites leaking the same two classes of PII: a user-supplied message
payload (reply text / STT transcript) and a user identifier (email / phone
number) that Vercel's retained, operator-viewable log stream should never
see in plaintext.

Named sites (telegram no-token reply, discord no-app-id reply, identity_store
unverified-marker skip, twilio STT transcript) plus two more identity_store
sites found by the sweep (link_channel, and the concurrent-mint-race
fallback) and two more twilio_voice sites (the invite-only rejection log and
the stream-started log both print the caller's raw phone number).

Every test drives the real function/coroutine (not a source grep) and
asserts the plaintext PII is absent from captured stdout.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import unittest
from unittest.mock import AsyncMock, patch

import audioop
import pytest
from fastapi import WebSocketDisconnect


# --- 1. telegram_webhook._send_reply leaks the reply text -----------------

def test_telegram_no_token_does_not_print_reply_text(capsys):
    from apps.channels.telegram_webhook import _send_reply

    secret_text = "your SSN on file is 123-45-6789"
    with patch("apps.channels.telegram_webhook.settings") as s:
        s.telegram_bot_token = None
        asyncio.run(_send_reply(12345, secret_text))

    out = capsys.readouterr().out
    assert secret_text not in out, f"reply text leaked into stdout: {out!r}"


# --- 2. discord._send_followup leaks the reply text ------------------------

def test_discord_no_app_id_does_not_print_reply_text(capsys):
    from apps.channels.discord import _send_followup

    secret_text = "your one-time login code is 048213"
    with patch("apps.channels.discord.settings") as s:
        s.discord_application_id = None
        asyncio.run(_send_followup("some-interaction-token", secret_text))

    out = capsys.readouterr().out
    assert secret_text not in out, f"reply text leaked into stdout: {out!r}"


# --- 3/5/6. identity_store: unverified-marker skip, link_channel, mint race

class IdentityStoreEmailLeakTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
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

    async def test_unverified_marker_skip_does_not_print_plaintext_email(self) -> None:
        email = "bob.unverified@example.com"
        self.pool.seed_marker("usr_other", email, verified=False)

        with _capture_stdout() as buf:
            user_id = await self.manager.resolve_user("web_email", email)

        # Unverified marker must not rebind -> mints a fresh user instead.
        self.assertTrue(user_id.startswith("usr_"))
        self.assertNotEqual(user_id, "usr_other")
        output = buf.getvalue()
        assert email not in output, f"plaintext email leaked into stdout: {output!r}"

    async def test_link_channel_does_not_print_plaintext_email(self) -> None:
        email = "carol.linked@example.com"
        with _capture_stdout() as buf:
            await self.manager.link_channel(
                user_id="usr_target", channel="web_email", channel_user_id=email
            )

        output = buf.getvalue()
        assert email not in output, f"plaintext email leaked into stdout: {output!r}"

    async def test_mint_race_does_not_print_plaintext_email(self) -> None:
        import asyncpg

        from test.identity_store_cloud_test import _FakeAsyncConn

        email = "dana.race@example.com"

        # Shared across every acquire()'d connection (resolve_user opens
        # several) so the SELECT-call sequence is tracked process-wide, not
        # per-connection: call #1 is resolve_user's own pre-race
        # lookup_user check (must miss — the race hasn't landed yet), call
        # #2 is the post-conflict fallback SELECT that discovers the winner.
        select_calls = {"n": 0}

        class _RaceConn(_FakeAsyncConn):
            """Simulates two concurrent first-logins racing to mint: the
            plain (non ON CONFLICT) channel_mappings insert raises a unique
            violation because another request already won."""

            async def execute(self, sql: str, *args):
                sql_norm = " ".join(sql.split())
                if sql_norm == (
                    "INSERT INTO channel_mappings (channel, channel_user_id, user_id) "
                    "VALUES ($1, $2, $3)"
                ):
                    raise asyncpg.UniqueViolationError("duplicate")
                return await super().execute(sql, *args)

            async def fetchrow(self, sql: str, *args):
                sql_norm = " ".join(sql.split())
                if sql_norm.startswith("SELECT user_id FROM channel_mappings"):
                    select_calls["n"] += 1
                    if select_calls["n"] == 1:
                        return None
                return await super().fetchrow(sql, *args)

        # Seed the "winner" mapping so the post-race fallback SELECT finds
        # it — but NOT visible to resolve_user's own first lookup (above).
        self.pool._db["channel_mappings"][("web_email", email)] = "usr_winner"

        self.pool.acquire = lambda: _FakeAcquireCtxFor(_RaceConn(self.pool._db))

        with _capture_stdout() as buf:
            user_id = await self.manager.resolve_user("web_email", email)

        self.assertEqual(user_id, "usr_winner")
        output = buf.getvalue()
        assert email not in output, f"plaintext email leaked into stdout: {output!r}"


class _FakeAcquireCtxFor:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


# --- helpers ---------------------------------------------------------------

import contextlib
import io


@contextlib.contextmanager
def _capture_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# --- 4/7/8. twilio_voice: STT transcript + caller phone number -------------

def _loud_ulaw_frame() -> bytes:
    """A synthetic PCMU frame well above the silence-energy threshold."""
    pcm = struct.pack("<160h", *([20000] * 160))
    return audioop.lin2ulaw(pcm, 2)


class _FakeTwilioWS:
    def __init__(self, incoming: list[str]):
        self._incoming = list(incoming)
        self.sent: list[str] = []

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if not self._incoming:
            raise WebSocketDisconnect()
        return self._incoming.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self) -> None:
        pass


def test_twilio_voice_stream_does_not_print_stt_transcript(capsys):
    from apps.channels import twilio_voice as tv

    secret_transcript = "my bank account number is 4111 1111 1111 1111"
    messages = [
        json.dumps({
            "event": "start",
            "start": {
                "callSid": "CA_test_transcript",
                "customParameters": {"from": "+15551230000"},
            },
        }),
        json.dumps({
            "event": "media",
            "media": {"payload": base64.b64encode(_loud_ulaw_frame()).decode()},
        }),
        json.dumps({"event": "stop"}),
    ]
    ws = _FakeTwilioWS(messages)

    with patch(
        "apps.channels.twilio_voice.resolve_channel_user", new=AsyncMock(return_value="usr_test")
    ), patch(
        "apps.channels.twilio_voice._transcribe_audio", new=AsyncMock(return_value=secret_transcript)
    ), patch(
        "apps.channels.twilio_voice.dispatch_event",
        new=AsyncMock(return_value=tv.TurtleResponse(content="ok", channel="twilio_voice", user_id="usr_test")),
    ), patch(
        "apps.channels.twilio_voice._synthesize_ulaw", new=AsyncMock(return_value=b"")
    ):
        asyncio.run(tv.voice_stream(ws))

    out = capsys.readouterr().out
    assert secret_transcript not in out, f"STT transcript leaked into stdout: {out!r}"


def test_twilio_voice_stream_started_does_not_print_caller_number(capsys):
    from apps.channels import twilio_voice as tv

    phone = "+15559876543"
    messages = [
        json.dumps({
            "event": "start",
            "start": {
                "callSid": "CA_test_started",
                "customParameters": {"from": phone},
            },
        }),
        json.dumps({"event": "stop"}),
    ]
    ws = _FakeTwilioWS(messages)

    with patch(
        "apps.channels.twilio_voice.resolve_channel_user", new=AsyncMock(return_value="usr_test")
    ):
        asyncio.run(tv.voice_stream(ws))

    out = capsys.readouterr().out
    assert phone not in out, f"caller phone number leaked into stdout: {out!r}"


def test_twilio_voice_invite_rejection_does_not_print_caller_number(capsys):
    from apps.channels import twilio_voice as tv

    phone = "+15551112222"
    messages = [
        json.dumps({
            "event": "start",
            "start": {
                "callSid": "CA_test_reject",
                "customParameters": {"from": phone},
            },
        }),
    ]
    ws = _FakeTwilioWS(messages)

    with patch(
        "apps.channels.twilio_voice.resolve_channel_user", new=AsyncMock(return_value=None)
    ), patch(
        "apps.channels.twilio_voice._synthesize_ulaw", new=AsyncMock(return_value=b"")
    ):
        asyncio.run(tv.voice_stream(ws))

    out = capsys.readouterr().out
    assert phone not in out, f"caller phone number leaked into stdout: {out!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
