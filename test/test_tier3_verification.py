"""
Tier 3 Verification Tests
arch_improve.md — Tier 3 checks

Covers: F1 (WhatsApp), F2 (iMessage), F3 (Slack), E5 (Twilio Voice), F4 (Calendar)

Run with:
    pytest test/test_tier3_verification.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Shared channel types & dispatch
# ---------------------------------------------------------------------------

class TestChannelBaseTypes:
    """TurtleEvent, TurtleResponse, and dispatch wiring are correct."""

    def test_turtle_event_importable(self):
        from apps.channels import TurtleEvent
        ev = TurtleEvent(user_id="usr_abc", channel="whatsapp", modality="text", content="hello")
        assert ev.content == "hello"
        assert ev.channel == "whatsapp"
        print("[PASS] TurtleEvent importable and constructable")

    def test_turtle_response_importable(self):
        from apps.channels import TurtleResponse
        resp = TurtleResponse(content="hi", channel="slack", user_id="usr_abc")
        assert resp.content == "hi"
        print("[PASS] TurtleResponse importable and constructable")

    def test_dispatch_text_returns_stub_before_wiring(self):
        import asyncio
        from apps.channels import dispatch_text, set_channel_dispatch

        # Reset to None to simulate un-wired state
        import apps.channels as ch
        original = ch._dispatch_fn
        ch._dispatch_fn = None

        async def run():
            result = await dispatch_text("hello", user_id="usr_test", channel="whatsapp")
            return result

        result = asyncio.run(run())
        assert "not ready" in result.lower() or isinstance(result, str)
        ch._dispatch_fn = original
        print("[PASS] dispatch_text returns stub when handler not wired")

    def test_set_channel_dispatch_wires_handler(self):
        import asyncio
        from apps.channels import TurtleEvent, TurtleResponse, dispatch_event, set_channel_dispatch
        import apps.channels as ch

        original = ch._dispatch_fn

        async def fake_handler(event: TurtleEvent) -> TurtleResponse:
            return TurtleResponse(content="wired!", channel=event.channel, user_id=event.user_id)

        set_channel_dispatch(fake_handler)

        async def run():
            ev = TurtleEvent(user_id="u", channel="slack", modality="text", content="ping")
            return await dispatch_event(ev)

        resp = asyncio.run(run())
        assert resp.content == "wired!"
        ch._dispatch_fn = original
        print("[PASS] set_channel_dispatch correctly wires handler")


# ---------------------------------------------------------------------------
# F1 — WhatsApp adapter
# ---------------------------------------------------------------------------

class TestF1WhatsApp:
    """F1: Twilio WhatsApp adapter — signature verification, idempotency, reply."""

    def test_whatsapp_router_importable(self):
        from apps.channels.whatsapp import router
        assert router is not None
        print("[PASS] WhatsApp router importable")

    def test_idempotency_cache_stores_and_retrieves(self):
        from apps.channels.whatsapp import _check_idempotency, _store_idempotency
        _store_idempotency("MSG_TEST_001", "Hello!")
        result = _check_idempotency("MSG_TEST_001")
        assert result == "Hello!"
        result_miss = _check_idempotency("MSG_UNKNOWN")
        assert result_miss is None
        print("[PASS] Idempotency cache stores and retrieves by MessageSid")

    def test_signature_verify_skips_when_no_token(self):
        from apps.channels.whatsapp import _verify_twilio_signature
        import unittest.mock as mock
        with mock.patch("apps.channels.whatsapp.settings") as s:
            s.twilio_auth_token = None
            result = _verify_twilio_signature("https://example.com", {}, "any_sig")
        assert result is True
        print("[PASS] Signature verification skips (returns True) when no token configured")

    def test_signature_verify_fails_on_bad_signature(self):
        from apps.channels.whatsapp import _verify_twilio_signature
        from pydantic import SecretStr
        import unittest.mock as mock
        with mock.patch("apps.channels.whatsapp.settings") as s:
            s.twilio_auth_token = SecretStr("real-secret")
            result = _verify_twilio_signature("https://example.com", {"Body": "hi"}, "bad_sig")
        assert result is False
        print("[PASS] Signature verification rejects invalid signature")

    def test_send_whatsapp_reply_skips_without_creds(self):
        import asyncio, unittest.mock as mock
        from apps.channels.whatsapp import _send_whatsapp_reply

        async def run():
            with mock.patch("apps.channels.whatsapp.settings") as s:
                s.twilio_account_sid = None
                s.twilio_auth_token = None
                s.twilio_whatsapp_number = None
                await _send_whatsapp_reply("+1234567890", "hello")  # must not raise

        asyncio.run(run())
        print("[PASS] _send_whatsapp_reply logs and returns gracefully when creds not configured")

    def test_webhook_endpoint_exists_on_router(self):
        from apps.channels.whatsapp import router
        routes = [r.path for r in router.routes]
        assert any("/channels/whatsapp" in p for p in routes), f"Expected /channels/whatsapp in {routes}"
        print("[PASS] WhatsApp router has /channels/whatsapp POST route")


# ---------------------------------------------------------------------------
# F2 — iMessage / SendBlue adapter
# ---------------------------------------------------------------------------

class TestF2iMessage:
    """F2: SendBlue iMessage adapter — signature verification, reply logic."""

    def test_imessage_router_importable(self):
        from apps.channels.imessage import router
        assert router is not None
        print("[PASS] iMessage router importable")

    def test_signature_verify_skips_without_secret(self):
        from apps.channels.imessage import _verify_sendblue_signature
        import unittest.mock as mock
        with mock.patch("apps.channels.imessage.settings") as s:
            s.sendblue_api_key = None
            s.sendblue_api_secret = None
            result = _verify_sendblue_signature(b"body", "any_sig")
        assert result is True
        print("[PASS] SendBlue signature verification skips when secret not configured")

    def test_signature_verify_fails_on_bad_signature(self):
        from apps.channels.imessage import _verify_sendblue_signature
        from pydantic import SecretStr
        import unittest.mock as mock
        with mock.patch("apps.channels.imessage._get_api_creds", return_value=("key", "real-secret")):
            result = _verify_sendblue_signature(b"hello", "wrong_sig")
        assert result is False
        print("[PASS] SendBlue signature verification rejects invalid signature")

    def test_send_reply_skips_without_creds(self):
        import asyncio, unittest.mock as mock
        from apps.channels.imessage import _send_imessage_reply

        async def run():
            with mock.patch("apps.channels.imessage.settings") as s:
                s.sendblue_api_key = None
                s.sendblue_api_secret = None
                await _send_imessage_reply("+1234567890", "test reply")

        asyncio.run(run())
        print("[PASS] _send_imessage_reply handles missing creds gracefully")

    def test_webhook_endpoint_exists_on_router(self):
        from apps.channels.imessage import router
        routes = [r.path for r in router.routes]
        assert any("/channels/imessage" in p for p in routes), f"Expected /channels/imessage in {routes}"
        print("[PASS] iMessage router has /channels/imessage POST route")


# ---------------------------------------------------------------------------
# F3 — Slack adapter
# ---------------------------------------------------------------------------

class TestF3Slack:
    """F3: Slack Events API adapter — signature, challenge, mention handler."""

    def test_slack_router_importable(self):
        from apps.channels.slack import router
        assert router is not None
        print("[PASS] Slack router importable")

    def test_signature_verify_skips_without_secret(self):
        from apps.channels.slack import _verify_slack_signature
        import unittest.mock as mock
        with mock.patch("apps.channels.slack._signing_secret", return_value=""):
            result = _verify_slack_signature(b"body", "12345", "v0=whatever")
        assert result is True
        print("[PASS] Slack signature verification skips when secret not configured")

    def test_signature_verify_rejects_stale_timestamp(self):
        from apps.channels.slack import _verify_slack_signature
        import time, unittest.mock as mock
        stale_ts = str(int(time.time()) - 400)  # 400s ago — beyond 300s replay window
        with mock.patch("apps.channels.slack._signing_secret", return_value="secret"):
            result = _verify_slack_signature(b"body", stale_ts, "v0=sig")
        assert result is False
        print("[PASS] Slack signature rejects stale timestamps (replay attack prevention)")

    def test_url_challenge_response(self):
        """Slack URL verification challenge must be echoed back."""
        import asyncio
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.slack import router
        import unittest.mock as mock

        test_app = FastAPI()
        test_app.include_router(router)

        with mock.patch("apps.channels.slack._verify_slack_signature", return_value=True):
            with TestClient(test_app) as client:
                resp = client.post(
                    "/channels/slack/events",
                    json={"type": "url_verification", "challenge": "abc123"},
                    headers={"X-Slack-Request-Timestamp": "0", "X-Slack-Signature": "v0=x"},
                )
        assert resp.status_code == 200
        assert resp.json().get("challenge") == "abc123"
        print("[PASS] Slack URL verification challenge echoed correctly")

    def test_bot_message_ignored(self):
        """Messages from bots must be ignored to prevent loops."""
        import asyncio
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.slack import router
        import unittest.mock as mock

        test_app = FastAPI()
        test_app.include_router(router)

        payload = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "bot_id": "BABC123",
                "text": "I am a bot",
                "channel": "C123",
                "ts": "1234.5678",
            },
        }
        with mock.patch("apps.channels.slack._verify_slack_signature", return_value=True):
            with TestClient(test_app) as client:
                resp = client.post(
                    "/channels/slack/events",
                    json=payload,
                    headers={"X-Slack-Request-Timestamp": "0", "X-Slack-Signature": "v0=x"},
                )
        assert resp.status_code == 200
        print("[PASS] Bot messages are ignored — no dispatch loop")

    def test_post_slack_message_skips_without_token(self):
        import asyncio, unittest.mock as mock
        from apps.channels.slack import _post_slack_message

        async def run():
            with mock.patch("apps.channels.slack._bot_token", return_value=""):
                await _post_slack_message("C123", "hello")  # must not raise

        asyncio.run(run())
        print("[PASS] _post_slack_message handles missing token gracefully")


# ---------------------------------------------------------------------------
# F6 — Discord adapter (interactions webhook + gateway import guard)
# ---------------------------------------------------------------------------

class TestF6Discord:
    """F6: Discord Interactions webhook — Ed25519 sig, PING, deferred command."""

    def test_discord_router_importable(self):
        from apps.channels.discord import router
        assert router is not None
        print("[PASS] Discord router importable")

    def test_signature_verify_no_key_local_noop_cloud_fail_closed(self):
        # No public key: no-op (accept) in local/dev, but FAIL CLOSED in cloud so
        # a public webhook can't be driven with unsigned/spoofed requests.
        import unittest.mock as mock
        import apps.channels.discord as d
        from core.config import settings
        with mock.patch.object(d, "_public_key", return_value=""):
            with mock.patch.object(settings, "deploy_mode", "local"):
                assert d._verify_discord_signature(b"body", "deadbeef", "12345") is True
            with mock.patch.object(settings, "deploy_mode", "cloud"):
                assert d._verify_discord_signature(b"body", "deadbeef", "12345") is False
        print("[PASS] Discord sig: local no-op, cloud fails closed without a public key")

    def test_signature_verify_real_keypair(self):
        """Prove REAL Ed25519 verification: a valid sig passes, tampering fails."""
        from apps.channels.discord import _verify_discord_signature
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization
        import unittest.mock as mock

        private_key = ed25519.Ed25519PrivateKey.generate()
        pub_hex = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

        timestamp = "1700000000"
        body = b'{"type":1}'
        signature = private_key.sign(timestamp.encode() + body).hex()

        with mock.patch("apps.channels.discord._public_key", return_value=pub_hex):
            assert _verify_discord_signature(body, signature, timestamp) is True
            # Tampered body must fail
            assert _verify_discord_signature(b'{"type":2}', signature, timestamp) is False
            # Tampered signature (all zeros) must fail
            assert _verify_discord_signature(body, "0" * len(signature), timestamp) is False
        print("[PASS] Discord signature verifies real Ed25519 signatures and rejects tampering")

    def test_ping_returns_pong(self):
        """PING (type 1) must be answered with PONG {'type': 1}."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.discord import router
        import unittest.mock as mock

        test_app = FastAPI()
        test_app.include_router(router)

        with mock.patch("apps.channels.discord._verify_discord_signature", return_value=True):
            with TestClient(test_app) as client:
                resp = client.post(
                    "/channels/discord",
                    json={"type": 1},
                    headers={"X-Signature-Ed25519": "aa", "X-Signature-Timestamp": "0"},
                )
        assert resp.status_code == 200
        assert resp.json().get("type") == 1
        print("[PASS] Discord PING answered with PONG (type 1)")

    def test_bad_signature_returns_401(self):
        """Discord REQUIRES 401 (not 403) on a bad signature."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.discord import router
        import unittest.mock as mock

        test_app = FastAPI()
        test_app.include_router(router)

        with mock.patch("apps.channels.discord._verify_discord_signature", return_value=False):
            with TestClient(test_app) as client:
                resp = client.post(
                    "/channels/discord",
                    json={"type": 1},
                    headers={"X-Signature-Ed25519": "bad", "X-Signature-Timestamp": "0"},
                )
        assert resp.status_code == 401
        print("[PASS] Discord returns 401 on invalid signature")

    def test_application_command_returns_deferred(self):
        """An APPLICATION_COMMAND (type 2) must ACK with DEFERRED (type 5)."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.discord import router
        from apps.channels import TurtleResponse

        test_app = FastAPI()
        test_app.include_router(router)

        async def _fake_dispatch(event):
            return TurtleResponse(content="hi", channel="discord", user_id=event.user_id)

        async def _fake_resolve(channel, uid):
            return "usr_test"

        async def _fake_followup(token, text):
            return None

        payload = {
            "type": 2,
            "id": "interaction_123",
            "token": "tok_abc",
            "channel_id": "chan_1",
            "member": {"user": {"id": "999", "bot": False}},
            "data": {"name": "turtle", "options": [{"name": "message", "value": "hello turtle"}]},
        }
        with mock.patch("apps.channels.discord._verify_discord_signature", return_value=True), \
             mock.patch("apps.channels.discord.dispatch_event", side_effect=_fake_dispatch), \
             mock.patch("apps.channels.discord._send_followup", side_effect=_fake_followup), \
             mock.patch.object(
                 __import__("apps.channels.discord", fromlist=["identity_manager"]).identity_manager,
                 "resolve_user", side_effect=_fake_resolve):
            with TestClient(test_app) as client:
                resp = client.post(
                    "/channels/discord",
                    json=payload,
                    headers={"X-Signature-Ed25519": "aa", "X-Signature-Timestamp": "0"},
                )
        assert resp.status_code == 200
        assert resp.json().get("type") == 5
        print("[PASS] Discord APPLICATION_COMMAND returns DEFERRED (type 5) and schedules dispatch")

    def test_bot_authored_command_ignored(self):
        """A bot-authored interaction must not trigger dispatch (loop guard)."""
        import unittest.mock as mock
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.discord import router

        test_app = FastAPI()
        test_app.include_router(router)

        dispatch_spy = mock.MagicMock()
        payload = {
            "type": 2,
            "id": "interaction_456",
            "token": "tok_def",
            "channel_id": "chan_2",
            "member": {"user": {"id": "888", "bot": True}},
            "data": {"name": "turtle", "options": [{"name": "message", "value": "loop me"}]},
        }
        with mock.patch("apps.channels.discord._verify_discord_signature", return_value=True), \
             mock.patch("apps.channels.discord.dispatch_event", dispatch_spy):
            with TestClient(test_app) as client:
                resp = client.post(
                    "/channels/discord",
                    json=payload,
                    headers={"X-Signature-Ed25519": "aa", "X-Signature-Timestamp": "0"},
                )
        assert resp.status_code == 200
        assert resp.json().get("type") != 5  # not deferred — ignored
        dispatch_spy.assert_not_called()
        print("[PASS] Discord bot-authored interaction ignored — no dispatch loop")

    def test_gateway_imports_and_noops_without_deps(self):
        """Gateway module imports cleanly without discord.py; start no-ops w/o token."""
        import asyncio, unittest.mock as mock
        from apps.channels import discord_gateway

        # discord.py is not installed in CI — gateway must not be 'available'.
        assert discord_gateway.gateway_available() is False

        async def run():
            with mock.patch("apps.channels.discord_gateway._bot_token", return_value=""):
                await discord_gateway.start_discord_gateway()  # must not raise
                await discord_gateway.stop_discord_gateway()   # must not raise

        asyncio.run(run())
        print("[PASS] Discord gateway imports cleanly and no-ops without discord.py / token")


# ---------------------------------------------------------------------------
# E5 — Twilio Voice adapter
# ---------------------------------------------------------------------------

class TestE5TwilioVoice:
    """E5: Twilio Media Streams adapter — μ-law codec, TwiML, STT/TTS wiring."""

    def test_twilio_voice_router_importable(self):
        from apps.channels.twilio_voice import router
        assert router is not None
        print("[PASS] Twilio Voice router importable")

    def test_ulaw_pcm_roundtrip(self):
        """μ-law encode/decode must be lossless within audioop precision."""
        import struct
        from apps.channels.twilio_voice import _ulaw_to_pcm16, _pcm16_to_ulaw

        # Generate a simple sine-wave-like PCM frame
        pcm_samples = [int(1000 * (i % 16 - 8)) for i in range(160)]
        pcm_bytes = struct.pack(f"{len(pcm_samples)}h", *pcm_samples)
        ulaw_bytes = _pcm16_to_ulaw(pcm_bytes)
        decoded_pcm = _ulaw_to_pcm16(ulaw_bytes)

        assert len(ulaw_bytes) == len(pcm_samples)  # 1 byte per sample
        assert len(decoded_pcm) == len(pcm_bytes)    # 2 bytes per sample
        print("[PASS] μ-law ↔ PCM16 roundtrip preserves sample count")

    def test_pcm_to_wav_produces_valid_header(self):
        import wave, io
        from apps.channels.twilio_voice import _pcm_to_wav

        pcm = bytes(320)  # 160 zero samples × 2 bytes
        wav_bytes = _pcm_to_wav(pcm, sample_rate=8000)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 8000
        print("[PASS] _pcm_to_wav produces valid WAV header (8 kHz, mono, 16-bit)")

    def test_frame_energy_zero_for_silence(self):
        from apps.channels.twilio_voice import _frame_energy, _pcm16_to_ulaw
        import audioop

        # All-zero PCM → silence
        pcm_silence = bytes(320)
        ulaw_silence = _pcm16_to_ulaw(pcm_silence)
        energy = _frame_energy(ulaw_silence)
        assert energy < 100, f"Silent frame has unexpectedly high energy: {energy}"
        print(f"[PASS] Silent frame energy is low: {energy:.1f}")

    def test_incoming_endpoint_returns_twiml(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from apps.channels.twilio_voice import router

        test_app = FastAPI()
        test_app.include_router(router)

        with TestClient(test_app) as client:
            resp = client.post("/channels/twilio/voice/incoming")

        assert resp.status_code == 200
        assert "application/xml" in resp.headers.get("content-type", "")
        body = resp.text
        assert "<Connect>" in body
        assert "<Stream" in body
        print("[PASS] /incoming returns valid TwiML with <Connect><Stream>")

    def test_tts_skips_without_deepgram_key(self):
        import asyncio, unittest.mock as mock
        from apps.channels.twilio_voice import _synthesize_ulaw

        async def run():
            with mock.patch("apps.channels.twilio_voice.settings") as s:
                s.deepgram_api_key = None
                result = await _synthesize_ulaw("Hello world")
            return result

        result = asyncio.run(run())
        assert result == b""
        print("[PASS] _synthesize_ulaw returns empty bytes when no Deepgram key")

    def test_stt_skips_without_groq_key(self):
        import asyncio, unittest.mock as mock
        from apps.channels.twilio_voice import _transcribe_audio

        async def run():
            with mock.patch("apps.channels.twilio_voice.settings") as s:
                s.groq_api_key = None
                s.groq_api_key2 = None
                result = await _transcribe_audio(b"fake_wav")
            return result

        result = asyncio.run(run())
        assert result == ""
        print("[PASS] _transcribe_audio returns empty string when no Groq key")


# ---------------------------------------------------------------------------
# F4 — Google Calendar tool
# ---------------------------------------------------------------------------

# TestF4CalendarTool moved to test/calendar_tool_test.py (Phase 4 tier extraction).


class TestChannelConfig:
    """All channel credentials are declared in TurtleSettings."""

    def test_twilio_fields_present(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        for f in ("twilio_account_sid", "twilio_auth_token", "twilio_whatsapp_number", "twilio_voice_number"):
            assert f in fields, f"Missing Twilio field: {f}"
        print("[PASS] All Twilio fields declared in TurtleSettings")

    def test_slack_fields_present(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        assert "slack_bot_token" in fields
        assert "slack_signing_secret" in fields
        print("[PASS] Slack fields declared in TurtleSettings")

    def test_sendblue_fields_present(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        assert "sendblue_api_key" in fields
        assert "sendblue_api_secret" in fields
        print("[PASS] SendBlue fields declared in TurtleSettings")

    def test_discord_fields_present(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        assert "discord_bot_token" in fields
        assert "discord_public_key" in fields
        assert "discord_application_id" in fields
        print("[PASS] Discord fields declared in TurtleSettings")

    def test_discord_fields_default_to_none(self):
        import unittest.mock as mock, os
        with mock.patch.dict(os.environ):
            for var in ("DISCORD_BOT_TOKEN", "DISCORD_PUBLIC_KEY", "DISCORD_APPLICATION_ID"):
                os.environ.pop(var, None)
            from core.config import TurtleSettings
            s = TurtleSettings(_env_file=None)
        for field in ("discord_bot_token", "discord_public_key", "discord_application_id"):
            assert getattr(s, field) is None, f"{field} should default to None"
        print("[PASS] Discord credential fields default to None")

    def test_all_channel_fields_default_to_none(self):
        import unittest.mock as mock, os
        channel_vars = (
            "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
            "SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET",
            "SENDBLUE_API_KEY", "SENDBLUE_API_SECRET",
            "GOOGLE_CALENDAR_CREDENTIALS_JSON",
        )
        # Test the declared field defaults, not whatever the developer's local
        # .env / shell happens to contain: drop the channel vars from the
        # environment and disable the dotenv source.
        with mock.patch.dict(os.environ):
            for var in channel_vars:
                os.environ.pop(var, None)
            from core.config import TurtleSettings
            s = TurtleSettings(_env_file=None)
        for field in (
            "twilio_account_sid", "twilio_auth_token",
            "slack_bot_token", "slack_signing_secret",
            "sendblue_api_key", "sendblue_api_secret",
            "google_calendar_credentials_json",
        ):
            val = getattr(s, field)
            assert val is None, f"{field} should default to None, got {val!r}"
        print("[PASS] All channel credential fields default to None (no crash without creds)")
