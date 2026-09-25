"""
test/wp1d_channel_signup_test.py
---------------------------------
WP 1.D (ledger 1a.4 / S-7.4) — channel sign-up policy + the channel-dispatch
rate limit.

Two independent pieces:

1. core.identity.resolve_channel_user: the sign-up-policy gate the 8 channel
   adapters call instead of identity_manager.resolve_user directly.
   TURTLE_CHANNEL_SIGNUP="open" (default) preserves today's behaviour
   (mint on miss); "invite" makes it a non-minting lookup. The 3 non-channel
   resolve_user callers (the re-resolve in _channel_dispatch_handler,
   apps/onboarding_routes.py, apps/auth.py) are NOT routed through the gate
   and must keep minting regardless of the policy.

2. The channel-dispatch rate limit in apps/turtle_server.py's
   _channel_dispatch_handler: reuses the same mode-aware limiter as the web
   WebSocket path, keyed on the raw (channel, channel_user_id) identity —
   never event.user_id — and runs before first-contact provisioning and
   before the per-(user, channel) lock.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import core.identity as identity_mod
from core.config import settings
from core.guardrails import WebSocketRateLimiter


# ---------------------------------------------------------------------------
# Fakes shared by the resolve_channel_user tests
# ---------------------------------------------------------------------------

class _FakeManager:
    """Mint-tracking stand-in for core.identity.identity_manager. Mirrors the
    real lookup_user/resolve_user contract (existing mapping wins; a miss
    mints for resolve_user, returns None for lookup_user) without touching
    disk, so the mint assertion is a hard fact (len(self.minted)), not a
    guess from reply text.
    """

    def __init__(self) -> None:
        self.mapping: dict[tuple[str, str], str] = {}
        self.minted: list[str] = []

    async def lookup_user(self, channel: str, channel_user_id: str):
        return self.mapping.get((channel, channel_user_id))

    async def resolve_user(self, channel: str, channel_user_id: str) -> str:
        key = (channel, channel_user_id)
        if key in self.mapping:
            return self.mapping[key]
        new_id = f"usr_{len(self.minted)}"
        self.mapping[key] = new_id
        self.minted.append(new_id)
        return new_id


@pytest.fixture
def fake_manager(monkeypatch):
    mgr = _FakeManager()
    monkeypatch.setattr(identity_mod, "identity_manager", mgr, raising=False)
    return mgr


# ---------------------------------------------------------------------------
# 1a. resolve_channel_user policy
# ---------------------------------------------------------------------------

def test_default_channel_signup_is_open():
    """The default must preserve today's behaviour — flipping an unconfigured
    deploy to invite-only would lock out every existing channel user."""
    assert settings.channel_signup == "open"


def test_open_default_mints_on_miss(fake_manager, monkeypatch):
    monkeypatch.setattr(settings, "channel_signup", "open")

    async def scenario():
        uid = await identity_mod.resolve_channel_user("discord", "unknown_123")
        assert uid is not None
        assert fake_manager.minted == [uid]

    asyncio.run(scenario())


def test_invite_unknown_sender_returns_none_and_mints_nothing(fake_manager, monkeypatch):
    monkeypatch.setattr(settings, "channel_signup", "invite")

    async def scenario():
        uid = await identity_mod.resolve_channel_user("discord", "unknown_123")
        assert uid is None
        assert fake_manager.minted == [], "invite-only must never mint on a miss"

    asyncio.run(scenario())


def test_invite_known_channel_user_is_unaffected(fake_manager, monkeypatch):
    fake_manager.mapping[("discord", "known_1")] = "usr_known"
    monkeypatch.setattr(settings, "channel_signup", "invite")

    async def scenario():
        uid = await identity_mod.resolve_channel_user("discord", "known_1")
        assert uid == "usr_known"
        assert fake_manager.minted == []

    asyncio.run(scenario())


def test_non_channel_callers_still_mint_under_invite(fake_manager, monkeypatch):
    """The 3 non-channel resolve_user callers (turtle_server's re-resolve,
    onboarding_routes.web_email, auth.web) call identity_manager.resolve_user
    directly, never resolve_channel_user, so the invite policy must not
    reach them."""
    monkeypatch.setattr(settings, "channel_signup", "invite")

    async def scenario():
        uid = await identity_mod.identity_manager.resolve_user("web_email", "new@example.com")
        assert uid is not None
        assert fake_manager.minted == [uid]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 1a-bis. normalize_channel_signup — a security toggle must never fail open
# on a typo (coordinator follow-up on WP 1.D). Covers casing, leading/
# trailing whitespace, an unrecognised value, and that unset still means
# open. Exercised at TWO levels: the pure function directly (asserts the
# canonical value AND the warn/silent behaviour), and through
# resolve_channel_user with settings.channel_signup set to the RAW
# (unnormalized) string via monkeypatch — matching how a real misconfigured
# env var would arrive — asserting the resulting mint/no-mint behaviour.
# ---------------------------------------------------------------------------

from core.config import (  # noqa: E402  (grouped with the rest of this section)
    CHANNEL_SIGNUP_INVITE,
    CHANNEL_SIGNUP_OPEN,
    normalize_channel_signup,
)


def test_normalize_unset_value_is_open_and_silent(capsys):
    assert normalize_channel_signup("") == CHANNEL_SIGNUP_OPEN
    assert normalize_channel_signup(None) == CHANNEL_SIGNUP_OPEN
    out = capsys.readouterr().out
    assert out == "", "an unset value is the expected default — it must not warn"


@pytest.mark.parametrize(
    "raw",
    ["INVITE", "Invite", "invite ", " invite", "  INVITE  "],
)
def test_normalize_case_and_whitespace_tolerant_invite(raw, capsys):
    assert normalize_channel_signup(raw) == CHANNEL_SIGNUP_INVITE
    out = capsys.readouterr().out
    assert out == "", f"a valid (if oddly-cased/padded) value must not warn: {raw!r}"


@pytest.mark.parametrize("raw", ["OPEN", "Open", "open ", " open"])
def test_normalize_case_and_whitespace_tolerant_open(raw, capsys):
    assert normalize_channel_signup(raw) == CHANNEL_SIGNUP_OPEN
    out = capsys.readouterr().out
    assert out == ""


def test_normalize_unrecognised_value_falls_back_to_open_and_warns(capsys):
    """The must-fix: an unrecognised value (a typo) must NOT silently behave
    as invite-only NOR silently behave as open — it falls back to open (the
    safe default direction) but is LOUD about it."""
    result = normalize_channel_signup("invyte")
    assert result == CHANNEL_SIGNUP_OPEN
    out = capsys.readouterr().out
    assert "invyte" in out
    assert "open" in out
    assert "invite" in out  # names the accepted values, not just the bad one


@pytest.mark.parametrize(
    "raw,expect_none",
    [
        ("INVITE", True),
        ("Invite", True),
        ("invite ", True),
        (" invite", True),
        ("invyte", False),  # unrecognised -> falls back to open -> mints
    ],
)
def test_resolve_channel_user_normalizes_raw_settings_value(
    raw, expect_none, fake_manager, monkeypatch
):
    """settings.channel_signup set to a RAW, unnormalized string (exactly
    what a misconfigured env var produces) must still gate correctly — this
    is the regression the coordinator's fail-open report was about."""
    monkeypatch.setattr(settings, "channel_signup", raw)

    async def scenario():
        uid = await identity_mod.resolve_channel_user("discord", "unknown_456")
        if expect_none:
            assert uid is None, f"{raw!r} must be treated as invite-only"
            assert fake_manager.minted == []
        else:
            assert uid is not None, f"{raw!r} (unrecognised) must fall back to open"
            assert fake_manager.minted == [uid]

    asyncio.run(scenario())


def test_channel_signup_field_validator_normalizes_at_construction():
    """The TurtleSettings field_validator is the startup-time half of the
    fix — env vars are read once at process boot, so this is where a real
    deployment's typo gets caught and logged."""
    from core.config import TurtleSettings

    s = TurtleSettings(TURTLE_CHANNEL_SIGNUP="  INVITE  ")
    assert s.channel_signup == CHANNEL_SIGNUP_INVITE

    s2 = TurtleSettings(TURTLE_CHANNEL_SIGNUP="invyte")
    assert s2.channel_signup == CHANNEL_SIGNUP_OPEN


# ---------------------------------------------------------------------------
# 1b. lookup_user on both identity backends
# ---------------------------------------------------------------------------

class TestLocalLookupUser:
    """Real aiosqlite IdentityManager — pins that lookup_user never mints and
    resolve_user's own mint behaviour is unchanged."""

    def test_lookup_miss_returns_none_and_mints_nothing(self, tmp_path):
        from core.identity import IdentityManager

        async def scenario():
            mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
            await mgr.init_db()
            assert await mgr.lookup_user("discord", "ghost") is None
            # No row was created by the lookup.
            assert await mgr.lookup_user("discord", "ghost") is None

        asyncio.run(scenario())

    def test_lookup_hit_after_resolve(self, tmp_path):
        from core.identity import IdentityManager

        async def scenario():
            mgr = IdentityManager(db_path=tmp_path / "users.sqlite")
            await mgr.init_db()
            minted = await mgr.resolve_user("discord", "759")
            found = await mgr.lookup_user("discord", "759")
            assert found == minted

        asyncio.run(scenario())


def test_cloud_lookup_user_mirrors_local_contract():
    """PostgresIdentityManager.lookup_user must behave identically to the
    local aiosqlite version: miss -> None, hit -> the mapped user_id. Uses
    the same fake-pool harness as test/identity_store_cloud_test.py (no live
    Postgres reachable in this environment)."""
    from unittest.mock import AsyncMock as _AsyncMock

    from core.storage.cloud.identity_store import PostgresIdentityManager

    class _FakeConn:
        def __init__(self, db):
            self._db = db

        async def execute(self, sql, *args):
            sql_norm = " ".join(sql.split())
            if sql_norm.startswith("INSERT INTO users"):
                self._db["users"].setdefault(args[0], {"primary_email": None})
            elif sql_norm.startswith("INSERT INTO channel_mappings"):
                channel, cuid, uid = args
                self._db["channel_mappings"][(channel, cuid)] = uid

        async def fetchrow(self, sql, *args):
            sql_norm = " ".join(sql.split())
            if sql_norm.startswith("SELECT user_id FROM channel_mappings"):
                channel, cuid = args
                uid = self._db["channel_mappings"].get((channel, cuid))
                return {"user_id": uid} if uid else None
            return None

        def transaction(self):
            class _Null:
                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, *exc):
                    return False

            return _Null()

    class _AcqCtx:
        def __init__(self, conn):
            self._conn = conn

        async def __aenter__(self):
            return self._conn

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def __init__(self):
            self._db = {"users": {}, "channel_mappings": {}}

        def acquire(self):
            return _AcqCtx(_FakeConn(self._db))

    pool = _Pool()
    with patch(
        "core.storage.cloud.identity_store.get_pg_pool",
        new_callable=_AsyncMock,
        return_value=pool,
    ):
        manager = PostgresIdentityManager()

        async def scenario():
            assert await manager.lookup_user("whatsapp", "+1555") is None
            minted = await manager.resolve_user("whatsapp", "+1555")
            found = await manager.lookup_user("whatsapp", "+1555")
            assert found == minted

        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 2. Channel-dispatch rate limit (apps/turtle_server.py)
# ---------------------------------------------------------------------------

class _FakeConfirmationGate:
    def next_prompt(self):
        return None

    def record_response(self, *args, **kwargs):
        pass


class _FakeSessionStore:
    message_history = None


class _FakeChannelState:
    """Minimal double for SharedState — only the attributes
    _channel_dispatch_handler actually touches once past the rate-limit
    check."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_store = _FakeSessionStore()
        self.confirmation_gate = _FakeConfirmationGate()
        self.channel_is_private = False
        self.channel = ""
        self.channel_user_id = ""
        self.http_client = None


@pytest.fixture
def dispatch_harness(monkeypatch):
    """Wires apps.turtle_server._channel_dispatch_handler to fakes cheap
    enough to run in a unit test: a same-user-id identity re-resolve, a
    trivial SharedState, and a canned _execute_turn outcome. The one piece
    NOT faked is the rate limiter itself.
    """
    import apps.turtle_server as ts

    async def _resolve_noop(channel, cuid):
        # Re-resolve is a no-op: return whatever user_id the caller already
        # has bound to this channel_user_id in this test (tracked via a dict
        # set on the fake itself).
        return fake_identity._bound.get((channel, cuid))

    fake_identity = SimpleNamespace(resolve_user=_resolve_noop, _bound={})

    monkeypatch.setattr(identity_mod, "identity_manager", fake_identity, raising=False)

    async def fake_build_channel_state(user_id, channel):
        return _FakeChannelState(user_id)

    execute_turn_mock = AsyncMock(return_value=SimpleNamespace(reply_text="ok", output_text="ok"))
    # provision_channel_user is a SYNC function run via asyncio.to_thread —
    # a plain (sync) mock, not AsyncMock, or to_thread would hand it a
    # never-awaited coroutine.
    from unittest.mock import MagicMock

    provision_mock = MagicMock(return_value=True)

    monkeypatch.setattr(ts, "_build_channel_state", fake_build_channel_state)
    monkeypatch.setattr(ts, "_execute_turn", execute_turn_mock)
    monkeypatch.setattr("core.user_provisioning.provision_channel_user", provision_mock)
    # Every test gets a private, uncontaminated cache + limiter.
    monkeypatch.setattr(ts, "_CHANNEL_STATES", {})

    return SimpleNamespace(
        ts=ts,
        fake_identity=fake_identity,
        execute_turn=execute_turn_mock,
        provision=provision_mock,
    )


def _event(ts, user_id, channel_user_id, channel="discord", sender_name=""):
    from apps.channels import TurtleEvent

    return TurtleEvent(
        user_id=user_id,
        channel=channel,
        modality="text",
        content="hi",
        message_id="m1",
        thread_id="t1",
        sender_name=sender_name,
        channel_user_id=channel_user_id,
    )


def test_rate_limit_refusal_is_a_deliverable_response_not_an_exception(dispatch_harness, monkeypatch):
    harness = dispatch_harness
    ts = harness.ts
    limiter = WebSocketRateLimiter(per_hour=1, per_day=100)
    monkeypatch.setattr(ts, "get_ws_rate_limiter", lambda: limiter)

    harness.fake_identity._bound[("discord", "chan_1")] = "usr_1"

    async def scenario():
        from apps.channels import TurtleResponse

        first = await ts._channel_dispatch_handler(_event(ts, "usr_1", "chan_1"))
        assert isinstance(first, TurtleResponse)
        assert first.content == "ok"

        second = await ts._channel_dispatch_handler(_event(ts, "usr_1", "chan_1"))
        assert isinstance(second, TurtleResponse)
        assert "try again" in second.content.lower() or "rate" in second.content.lower()
        # The second call never reached the pipeline.
        assert harness.execute_turn.call_count == 1

    asyncio.run(scenario())


def test_rate_limit_keyed_on_channel_identity_two_handles_counted_separately(dispatch_harness, monkeypatch):
    harness = dispatch_harness
    ts = harness.ts
    limiter = WebSocketRateLimiter(per_hour=1, per_day=100)
    monkeypatch.setattr(ts, "get_ws_rate_limiter", lambda: limiter)

    harness.fake_identity._bound[("discord", "chan_A")] = "usr_A"
    harness.fake_identity._bound[("discord", "chan_B")] = "usr_B"

    async def scenario():
        a1 = await ts._channel_dispatch_handler(_event(ts, "usr_A", "chan_A"))
        b1 = await ts._channel_dispatch_handler(_event(ts, "usr_B", "chan_B"))
        assert a1.content == "ok" and b1.content == "ok"
        assert harness.execute_turn.call_count == 2

        # chan_A is now at its cap; chan_B's own budget is untouched by it.
        a2 = await ts._channel_dispatch_handler(_event(ts, "usr_A", "chan_A"))
        assert "usr_A" not in a2.content  # sanity: not an echo
        assert harness.execute_turn.call_count == 2  # a2 was refused

    asyncio.run(scenario())


def test_rate_limit_keyed_on_channel_identity_not_event_user_id(dispatch_harness, monkeypatch):
    """A request whose event.user_id would be re-pointed by a link redemption
    is still counted against the raw channel identity, not the (possibly
    stale, possibly about-to-change) user_id."""
    harness = dispatch_harness
    ts = harness.ts
    limiter = WebSocketRateLimiter(per_hour=1, per_day=100)
    monkeypatch.setattr(ts, "get_ws_rate_limiter", lambda: limiter)

    harness.fake_identity._bound[("discord", "chan_X")] = "usr_stale"

    async def scenario():
        await ts._channel_dispatch_handler(_event(ts, "usr_stale", "chan_X"))

    asyncio.run(scenario())

    assert "discord:chan_X" in limiter._events
    assert "usr_stale" not in limiter._events


def test_rate_limit_runs_before_provisioning_and_lock(dispatch_harness, monkeypatch):
    """A refused request must cost no provisioning I/O — assert
    provision_channel_user was never invoked on the refused call."""
    harness = dispatch_harness
    ts = harness.ts
    limiter = WebSocketRateLimiter(per_hour=1, per_day=100)
    monkeypatch.setattr(ts, "get_ws_rate_limiter", lambda: limiter)

    harness.fake_identity._bound[("discord", "chan_1")] = "usr_1"

    async def scenario():
        await ts._channel_dispatch_handler(
            _event(ts, "usr_1", "chan_1", sender_name="Alice")
        )
        calls_after_first = harness.provision.call_count
        assert calls_after_first >= 1  # the allowed call DID provision

        await ts._channel_dispatch_handler(
            _event(ts, "usr_1", "chan_1", sender_name="Alice")
        )
        # The refused call must not have provisioned again.
        assert harness.provision.call_count == calls_after_first

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 3. Per-channel invite-only no-mint coverage.
#
# Discord (apps/channels/discord.py::_process_deferred_interaction) and
# Telegram-webhook (apps/channels/telegram_webhook.py::telegram_webhook) are
# covered in test/discord_deferred_processing_test.py and
# test/telegram_webhook_test.py respectively (each got a
# test_invite_only_unknown_sender_no_mint_no_dispatch case alongside this
# WP). The remaining 6 are here: imessage, whatsapp, slack, twilio_voice
# (webhooks) and discord_gateway / telegram_gateway (long-running gateway
# on_message closures).
#
# Every case patches resolve_channel_user directly (never real
# identity_manager) — this is the seam BOTH the "open" default and the
# "invite" gate route through: proving the adapter calls it and, on a None
# return, never reaches dispatch_event, is exactly "no tenant is minted" —
# dispatch_event is the only thing downstream that could turn a
# resolved/minted user_id into new state.
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_imessage_invite_only_unknown_sender_no_mint_no_dispatch():
    import apps.channels.imessage as im

    with patch.object(im, "_verify_sendblue_signature", return_value=True), patch.object(
        im, "resolve_channel_user", new_callable=AsyncMock, return_value=None
    ) as fake_resolve, patch.object(
        im, "dispatch_event", new_callable=AsyncMock
    ) as fake_dispatch, patch.object(
        im, "_send_imessage_reply", new_callable=AsyncMock
    ) as fake_send:
        client = _make_client(im.router)
        resp = client.post(
            "/channels/imessage",
            content=b'{"from_number": "+15551234567", "content": "hi", "message_handle": "h1"}',
            headers={"X-SendBlue-Signature": "sig"},
        )

    assert resp.status_code == 200
    fake_resolve.assert_awaited_once_with("imessage", "+15551234567")
    fake_dispatch.assert_not_called()
    fake_send.assert_awaited_once_with("+15551234567", im.CHANNEL_INVITE_ONLY_MESSAGE)


def test_whatsapp_invite_only_unknown_sender_no_mint_no_dispatch():
    import apps.channels.whatsapp as wa

    with patch.object(wa, "_verify_twilio_signature", return_value=True), patch.object(
        wa, "resolve_channel_user", new_callable=AsyncMock, return_value=None
    ) as fake_resolve, patch.object(
        wa, "dispatch_event", new_callable=AsyncMock
    ) as fake_dispatch, patch.object(
        wa, "_send_whatsapp_reply", new_callable=AsyncMock
    ) as fake_send:
        client = _make_client(wa.router)
        resp = client.post(
            "/channels/whatsapp",
            data={"From": "whatsapp:+15551234567", "Body": "hi", "MessageSid": "SM1"},
            headers={"X-Twilio-Signature": "sig"},
        )

    assert resp.status_code == 200
    fake_resolve.assert_awaited_once_with("whatsapp", "whatsapp:+15551234567")
    fake_dispatch.assert_not_called()
    fake_send.assert_awaited_once_with("whatsapp:+15551234567", wa.CHANNEL_INVITE_ONLY_MESSAGE)


def test_slack_invite_only_unknown_sender_no_mint_no_dispatch():
    import apps.channels.slack as sl

    payload = {
        "type": "event_callback",
        "event": {
            "type": "message",
            "user": "U_UNKNOWN",
            "text": "hi",
            "channel": "C123",
            "ts": "1234.5678",
        },
    }
    with patch.object(sl, "_verify_slack_signature", return_value=True), patch.object(
        sl, "resolve_channel_user", new_callable=AsyncMock, return_value=None
    ) as fake_resolve, patch.object(
        sl, "dispatch_event", new_callable=AsyncMock
    ) as fake_dispatch, patch.object(
        sl, "_post_slack_message", new_callable=AsyncMock
    ) as fake_post:
        client = _make_client(sl.router)
        resp = client.post(
            "/channels/slack/events",
            json=payload,
            headers={"X-Slack-Request-Timestamp": "0", "X-Slack-Signature": "v0=x"},
        )
        assert resp.status_code == 200
        # The event is processed as a fire-and-forget background task —
        # give the loop a beat to run it before asserting.
        import time as _time

        for _ in range(50):
            if fake_resolve.await_count:
                break
            _time.sleep(0.02)

    fake_resolve.assert_awaited_once_with("slack", "U_UNKNOWN")
    fake_dispatch.assert_not_called()
    fake_post.assert_awaited_once()
    assert fake_post.call_args.args[1] == sl.CHANNEL_INVITE_ONLY_MESSAGE


def test_twilio_voice_invite_only_unknown_caller_no_mint_no_dispatch():
    import apps.channels.twilio_voice as tv

    with patch.object(
        tv, "resolve_channel_user", new_callable=AsyncMock, return_value=None
    ) as fake_resolve, patch.object(
        tv, "dispatch_event", new_callable=AsyncMock
    ) as fake_dispatch, patch.object(
        tv, "_synthesize_ulaw", new_callable=AsyncMock, return_value=b"\x00" * 160
    ) as fake_tts:
        client = _make_client(tv.router)
        with client.websocket_connect("/channels/twilio/voice/stream") as ws:
            ws.send_text(
                '{"event": "start", "start": {"callSid": "CA1", '
                '"customParameters": {"from": "+15551234567"}}}'
            )
            ws.send_text('{"event": "stop"}')
            # Connection closes server-side after "stop"; draining is optional.

    fake_resolve.assert_awaited_once_with("twilio_voice", "+15551234567")
    fake_dispatch.assert_not_called()
    fake_tts.assert_awaited_once_with(tv.CHANNEL_INVITE_ONLY_MESSAGE)


# --- Gateway closures (discord_gateway, telegram_gateway) -------------------
#
# Both run as long-lived background loops (discord.Client / a python-
# telegram-bot Application), so unlike the webhook adapters there is no HTTP
# request/response to drive. Each start_*_gateway() registers its on_message
# handler on a real client/application object; here that object is faked
# just enough to (a) avoid any real network connection and (b) let the test
# grab the registered coroutine and call it directly with a synthetic event.

def test_discord_gateway_invite_only_unknown_sender_no_mint_no_dispatch(monkeypatch):
    import apps.channels.discord_gateway as dg

    class _FakeIntents:
        message_content = False
        dm_messages = False

        @classmethod
        def default(cls):
            return cls()

    class _FakeDMChannel:
        def __init__(self):
            self.send = AsyncMock()

    class _FakeClient:
        def __init__(self, intents=None):
            self.user = None
            self.intents = intents

        def event(self, coro):
            setattr(self, coro.__name__, coro)
            return coro

        async def start(self, token):
            return  # no real connection

    class _FakeDiscordModule:
        Client = _FakeClient
        Intents = _FakeIntents
        DMChannel = _FakeDMChannel

    monkeypatch.setattr(dg, "_client", None, raising=False)
    monkeypatch.setattr(dg, "_client_task", None, raising=False)
    monkeypatch.setattr(dg, "discord", _FakeDiscordModule)
    monkeypatch.setattr(dg.settings, "discord_bot_token", SimpleNamespace(get_secret_value=lambda: "fake-token"))

    fake_resolve = AsyncMock(return_value=None)
    fake_dispatch = AsyncMock()

    async def scenario():
        # These closures do their imports LOCALLY inside start_*_gateway
        # ("from apps.channels import ... dispatch_event" / "from
        # core.identity import ... resolve_channel_user"), binding at
        # call-time from the SOURCE modules — patch those, not the
        # gateway module's own (unused) namespace.
        with patch("apps.channels.dispatch_event", fake_dispatch), patch(
            "core.identity.resolve_channel_user", fake_resolve
        ):
            await dg.start_discord_gateway()
            client = dg._client
            assert client is not None

            channel = _FakeDMChannel()
            message = SimpleNamespace(
                content="hello",
                mentions=[],
                channel=channel,
                author=SimpleNamespace(
                    bot=False, id=999, display_name="Alice", global_name="", name="alice"
                ),
                id="m1",
            )
            await client.on_message(message)

            fake_resolve.assert_awaited_once_with("discord", "999")
            fake_dispatch.assert_not_called()
            channel.send.assert_awaited_once_with(identity_mod.CHANNEL_INVITE_ONLY_MESSAGE)

    asyncio.run(scenario())
    dg._client = None
    dg._client_task = None


def test_telegram_gateway_invite_only_unknown_sender_no_mint_no_dispatch(monkeypatch):
    import apps.channels.telegram_gateway as tg

    class _FakeUpdater:
        start_polling = AsyncMock()

    class _FakeApplication:
        def __init__(self):
            self.handlers = {0: []}
            self.bot = SimpleNamespace(username="turtlebot")
            self.updater = _FakeUpdater()

        def add_handler(self, handler):
            self.handlers[0].append(handler)

        async def initialize(self):
            return

        async def start(self):
            return

    class _FakeApplicationBuilder:
        def token(self, _t):
            return self

        def build(self):
            return _FakeApplication()

    monkeypatch.setattr(tg, "_app", None, raising=False)
    monkeypatch.setattr(tg, "_app_task", None, raising=False)
    monkeypatch.setattr(tg, "ApplicationBuilder", _FakeApplicationBuilder)
    monkeypatch.setattr(tg.settings, "telegram_bot_token", SimpleNamespace(get_secret_value=lambda: "fake-token"))

    fake_resolve = AsyncMock(return_value=None)
    fake_dispatch = AsyncMock()

    async def scenario():
        with patch("apps.channels.dispatch_event", fake_dispatch), patch(
            "core.identity.resolve_channel_user", fake_resolve
        ):
            await tg.start_telegram_gateway()
            app = tg._app
            assert app is not None
            on_message = app.handlers[0][0].callback

            message = SimpleNamespace(
                text="hello",
                caption=None,
                from_user=SimpleNamespace(id=555, is_bot=False, first_name="Bob", username="bob", full_name="Bob"),
                message_id=1,
                reply_text=AsyncMock(),
            )
            chat = SimpleNamespace(type="private", id=42)
            update = SimpleNamespace(effective_message=message, effective_chat=chat)
            context = SimpleNamespace(bot=SimpleNamespace(username="turtlebot"))

            await on_message(update, context)

            fake_resolve.assert_awaited_once_with("telegram", "555")
            fake_dispatch.assert_not_called()
            message.reply_text.assert_awaited_once_with(identity_mod.CHANNEL_INVITE_ONLY_MESSAGE)

    asyncio.run(scenario())
    tg._app = None
    tg._app_task = None
