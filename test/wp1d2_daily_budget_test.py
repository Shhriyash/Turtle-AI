"""
WP1.D2 (ledger 1a.4 part 4) — per-tenant daily token budget.

INCRBY turtle:spend:{uid}:{yyyymmdd}, EXPIRE 2 days, UTC day boundary.
Default 1,000,000 tokens/user/day (TURTLE_DAILY_TOKEN_BUDGET);
TURTLE_UNMETERED_USER_IDS exempt. Cloud-mode only — local has no Redis and is
never metered (see apps/turtle_server.py's module comment above TurnOutcome
for the full ordering/overshoot/Redis-failure reasoning these tests pin).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import apps.turtle_server as ts
from core.config import parse_unmetered_user_ids


class _FakePipeline:
    def __init__(self, client):
        self._client = client
        self._ops = []

    def incrby(self, key, amount):
        self._ops.append(("incrby", key, amount))
        return self

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds))
        return self

    def execute(self):
        for op, key, val in self._ops:
            if op == "incrby":
                self._client.store[key] = int(self._client.store.get(key, 0)) + val
            elif op == "expire":
                self._client.expiries[key] = val
        self._ops = []


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}
        self.expiries: dict[str, int] = {}

    def get(self, key):
        val = self.store.get(key)
        return str(val) if val is not None else None

    def pipeline(self):
        return _FakePipeline(self)


class _BrokenRedis:
    """Simulates Redis being unreachable."""

    def get(self, key):
        raise ConnectionError("simulated Redis outage")

    def pipeline(self):
        raise ConnectionError("simulated Redis outage")


@pytest.fixture()
def cloud_mode(monkeypatch):
    monkeypatch.setattr(ts.settings, "deploy_mode", "cloud", raising=False)
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1000, raising=False)
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "", raising=False)
    return None


def _patch_redis(monkeypatch, client):
    import core.storage.cloud as cloud_mod

    monkeypatch.setattr(cloud_mod, "get_redis_sync_client", lambda: client, raising=False)
    return client


# ── local mode: always unmetered ─────────────────────────────────────────────

def test_local_mode_never_refuses(monkeypatch):
    monkeypatch.setattr(ts.settings, "deploy_mode", "local", raising=False)
    monkeypatch.setattr(ts.settings, "daily_token_budget", 1, raising=False)  # absurdly tight
    assert ts._daily_spend_check("usr_anyone") is None
    ts._record_daily_spend("usr_anyone", 999999)  # must not raise, must not touch Redis


# ── cloud mode: check / refuse / reset time ─────────────────────────────────

def test_under_budget_proceeds(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_a")] = 500  # under the 1000 limit
    assert ts._daily_spend_check("usr_a") is None


def test_at_or_over_budget_is_refused_and_names_a_reset_time(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_a")] = 1000  # at the limit
    refusal = ts._daily_spend_check("usr_a")
    assert refusal is not None
    assert "resets at" in refusal
    # The reset time is a concrete UTC timestamp string, not a placeholder.
    reset_str = refusal.split("resets at", 1)[1].strip().rstrip(".")
    assert "UTC" in reset_str
    parsed = datetime.strptime(reset_str, "%H:%M UTC on %Y-%m-%d")
    now = datetime.now(UTC).replace(tzinfo=None)
    assert now < parsed <= now + timedelta(days=1, minutes=1)


def test_unmetered_user_is_never_refused_even_far_over_budget(cloud_mode, monkeypatch):
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "usr_owner, usr_other", raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_owner")] = 10_000_000
    assert ts._daily_spend_check("usr_owner") is None


def test_budget_disabled_when_zero_or_negative(cloud_mode, monkeypatch):
    monkeypatch.setattr(ts.settings, "daily_token_budget", 0, raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    client.store[ts._spend_key("usr_a")] = 999_999_999
    assert ts._daily_spend_check("usr_a") is None


# ── recording spend ──────────────────────────────────────────────────────────

def test_under_budget_records_spend_by_the_turns_tokens(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    key = ts._spend_key("usr_a")
    assert client.store.get(key) is None
    ts._record_daily_spend("usr_a", 250)
    assert client.store[key] == 250
    ts._record_daily_spend("usr_a", 100)
    assert client.store[key] == 350
    assert client.expiries[key] == 172800  # 2 days


def test_unmetered_user_spend_is_never_recorded(cloud_mode, monkeypatch):
    monkeypatch.setattr(ts.settings, "unmetered_user_ids", "usr_owner", raising=False)
    client = _patch_redis(monkeypatch, _FakeRedis())
    ts._record_daily_spend("usr_owner", 500)
    assert ts._spend_key("usr_owner") not in client.store


def test_zero_or_negative_tokens_do_not_touch_redis(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    ts._record_daily_spend("usr_a", 0)
    ts._record_daily_spend("usr_a", -5)
    assert client.store == {}


# ── UTC day boundary ─────────────────────────────────────────────────────────

def test_spend_key_rolls_at_utc_midnight(monkeypatch):
    import apps.turtle_server as ts_mod

    class _FixedDT:
        _now = datetime(2026, 9, 25, 23, 59, 59, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls._now

    monkeypatch.setattr("apps.turtle_server._utc_day_str", lambda: "20260925")
    key_before = ts_mod._spend_key("usr_a")
    monkeypatch.setattr("apps.turtle_server._utc_day_str", lambda: "20260926")
    key_after = ts_mod._spend_key("usr_a")
    assert key_before != key_after
    assert key_before == "turtle:spend:usr_a:20260925"
    assert key_after == "turtle:spend:usr_a:20260926"


def test_spend_from_yesterday_does_not_count_against_today(cloud_mode, monkeypatch):
    client = _patch_redis(monkeypatch, _FakeRedis())
    monkeypatch.setattr("apps.turtle_server._utc_day_str", lambda: "20260101")
    ts._record_daily_spend("usr_a", 999)  # yesterday, near/at limit
    monkeypatch.setattr("apps.turtle_server._utc_day_str", lambda: "20260102")
    # Today's key is fresh — under budget despite yesterday's near-max spend.
    assert ts._daily_spend_check("usr_a") is None


# ── Redis unavailable: fail OPEN (pinned posture) ────────────────────────────

def test_redis_unavailable_check_fails_open(cloud_mode, monkeypatch):
    _patch_redis(monkeypatch, _BrokenRedis())
    # Must not raise, and must allow the turn (fail open — see module comment
    # in apps/turtle_server.py for why this differs from wave 1's email
    # reservation fail-closed choice).
    assert ts._daily_spend_check("usr_a") is None


def test_redis_unavailable_record_does_not_raise(cloud_mode, monkeypatch):
    _patch_redis(monkeypatch, _BrokenRedis())
    ts._record_daily_spend("usr_a", 500)  # best-effort: swallow, log, move on


# ── per-turn UsageLimits ceiling ─────────────────────────────────────────────

def test_usage_limits_has_a_per_turn_token_ceiling():
    assert ts.agents_mgr.usage_limits.total_tokens_limit == 100_000
    assert ts.agents_mgr.usage_limits.request_limit == 30


# ── parse_unmetered_user_ids ─────────────────────────────────────────────────

def test_parse_unmetered_user_ids_trims_and_drops_empties():
    assert parse_unmetered_user_ids(" usr_a, usr_b ,,") == frozenset({"usr_a", "usr_b"})
    assert parse_unmetered_user_ids("") == frozenset()
    assert parse_unmetered_user_ids(None) == frozenset()
