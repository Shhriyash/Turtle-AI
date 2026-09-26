"""
test/account_linking_cloud_test.py
--------------------------------------
Unit coverage for core/storage/cloud/account_linking_store.py (post-migration
audit fix): identity_manager.db_path only exists on the LOCAL IdentityManager,
so the two production call sites building LinkCodeStore(identity_manager.db_path)
would raise AttributeError outright in cloud mode. core.storage.factory
.get_link_code_store() is the fix. Uses a lightweight fake psycopg pool — no
live Postgres reachable in this environment (same caveat as the rest of this
migration's unit tests).
"""
from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from core.storage.cloud.account_linking_store import PostgresLinkCodeStore


class _FakeCursor:
    def __init__(self, row, rowcount=0):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, table: dict):
        self.table = table  # code -> dict(channel, channel_user_id, source_user_id, expires_at, consumed_at, reserved_for, reserved_at, expected_email)

    def execute(self, sql: str, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("DELETE FROM link_codes WHERE channel"):
            channel, channel_user_id = params
            for code, row in list(self.table.items()):
                if row["channel"] == channel and row["channel_user_id"] == channel_user_id and row["consumed_at"] is None:
                    del self.table[code]
            return _FakeCursor(None)
        if sql_norm.startswith("INSERT INTO link_codes"):
            code, channel, channel_user_id, source_user_id, expires_at, expected_email = params
            self.table[code] = {
                "channel": channel, "channel_user_id": channel_user_id,
                "source_user_id": source_user_id, "expires_at": expires_at,
                "consumed_at": None, "reserved_for": None, "reserved_at": None,
                "expected_email": expected_email,
            }
            return _FakeCursor(None)
        if sql_norm.startswith("SELECT channel, channel_user_id, source_user_id, expires_at, consumed_at, expected_email FROM link_codes"):
            code = params[0]
            row = self.table.get(code)
            if row is None:
                return _FakeCursor(None)
            return _FakeCursor((
                row["channel"], row["channel_user_id"], row["source_user_id"],
                row["expires_at"], row["consumed_at"], row.get("expected_email"),
            ))
        if sql_norm.startswith("UPDATE link_codes SET reserved_for = NULL"):
            code, target_user_id = params
            row = self.table.get(code)
            if row and row.get("reserved_for") == target_user_id and row["consumed_at"] is None:
                row["reserved_for"] = None
                row["reserved_at"] = None
            return _FakeCursor(None)
        if sql_norm.startswith("UPDATE link_codes SET reserved_for"):
            target_user_id, now, code, now2, target_user_id2, cutoff = params
            row = self.table.get(code)
            matched = 0
            if row and row["consumed_at"] is None and row["expires_at"] > now2:
                if row["reserved_for"] is None or row["reserved_for"] == target_user_id2 or row["reserved_at"] is None or row["reserved_at"] < cutoff:
                    row["reserved_for"] = target_user_id
                    row["reserved_at"] = now
                    matched = 1
            return _FakeCursor(None, rowcount=matched)
        if sql_norm.startswith("UPDATE link_codes SET consumed_at"):
            consumed_at, code = params
            row = self.table.get(code)
            matched = 0
            if row and row["consumed_at"] is None:
                row["consumed_at"] = consumed_at
                matched = 1
            return _FakeCursor(None, rowcount=matched)
        if sql_norm.startswith("DELETE FROM link_codes WHERE expires_at"):
            cutoff = params[0]
            expired = [c for c, r in self.table.items() if r["expires_at"] <= cutoff]
            for c in expired:
                del self.table[c]
            return _FakeCursor(None, rowcount=len(expired))
        return _FakeCursor(None)  # CREATE TABLE


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.table: dict = {}

    def connection(self):
        return _FakeConnCtx(_FakeConn(self.table))


class PostgresLinkCodeStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = _FakePool()
        import core.storage.cloud.account_linking_store as als

        als._initialized = False
        patcher = patch(
            "core.storage.cloud.account_linking_store.get_pg_sync_pool", return_value=self.pool
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.store = PostgresLinkCodeStore()

    def test_issue_then_peek_round_trips(self) -> None:
        issued = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        self.assertEqual(len(issued.code), 8)
        peeked = self.store.peek(issued.code)
        self.assertIsNotNone(peeked)
        self.assertEqual(peeked.channel, "discord")
        self.assertEqual(peeked.source_user_id, "usr_src")
        self.assertIsNone(peeked.expected_email)

    def test_issue_normalizes_and_carries_expected_email(self) -> None:
        issued = self.store.issue(
            channel="discord", channel_user_id="123", source_user_id="usr_src",
            expected_email="  Me@Example.COM  ",
        )
        self.assertEqual(issued.expected_email, "me@example.com")
        peeked = self.store.peek(issued.code)
        self.assertEqual(peeked.expected_email, "me@example.com")
        status, claim = self.store.reserve(issued.code, "usr_target")
        self.assertEqual(status, "ok")
        self.assertEqual(claim.expected_email, "me@example.com")
        consumed = self.store.consume(issued.code)
        self.assertEqual(consumed.expected_email, "me@example.com")

    def test_issue_drops_previous_unconsumed_code_for_same_identity(self) -> None:
        first = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        second = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        self.assertIsNone(self.store.peek(first.code))
        self.assertIsNotNone(self.store.peek(second.code))

    def test_peek_unknown_code_returns_none(self) -> None:
        self.assertIsNone(self.store.peek("NOPE1234"))

    def test_peek_expired_code_returns_none(self) -> None:
        issued = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        self.pool.table[issued.code]["expires_at"] = datetime.now(UTC) - timedelta(minutes=1)
        self.assertIsNone(self.store.peek(issued.code))

    def test_reserve_then_consume_succeeds(self) -> None:
        issued = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        status, claim = self.store.reserve(issued.code, "usr_target")
        self.assertEqual(status, "ok")
        self.assertIsNotNone(claim)
        consumed = self.store.consume(issued.code)
        self.assertIsNotNone(consumed)
        self.assertIsNone(self.store.consume(issued.code))  # single-use

    def test_reserve_blocks_a_different_target_within_ttl(self) -> None:
        issued = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        first_status, _ = self.store.reserve(issued.code, "usr_a")
        second_status, second_claim = self.store.reserve(issued.code, "usr_b")
        self.assertEqual(first_status, "ok")
        self.assertEqual(second_status, "locked")
        self.assertIsNone(second_claim)

    def test_reserve_same_target_is_idempotent(self) -> None:
        issued = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        self.store.reserve(issued.code, "usr_a")
        status, claim = self.store.reserve(issued.code, "usr_a")
        self.assertEqual(status, "ok")
        self.assertIsNotNone(claim)

    def test_release_reservation_allows_a_different_target_to_reserve(self) -> None:
        issued = self.store.issue(channel="discord", channel_user_id="123", source_user_id="usr_src")
        self.store.reserve(issued.code, "usr_a")
        self.store.release_reservation(issued.code, "usr_a")
        status, _ = self.store.reserve(issued.code, "usr_b")
        self.assertEqual(status, "ok")

    def test_consume_unknown_code_returns_none(self) -> None:
        self.assertIsNone(self.store.consume("NOPE1234"))

    def test_purge_expired_removes_only_expired_codes(self) -> None:
        issued_live = self.store.issue(channel="discord", channel_user_id="1", source_user_id="usr_a")
        issued_expired = self.store.issue(channel="discord", channel_user_id="2", source_user_id="usr_b")
        self.pool.table[issued_expired.code]["expires_at"] = datetime.now(UTC) - timedelta(minutes=1)
        removed = self.store.purge_expired()
        self.assertEqual(removed, 1)
        self.assertIsNotNone(self.store.peek(issued_live.code))


class GetLinkCodeStoreFactoryTest(unittest.TestCase):
    def test_cloud_mode_returns_postgres_store(self) -> None:
        with patch("core.storage.factory.settings") as fake_settings:
            fake_settings.is_cloud = True
            from core.storage.factory import get_link_code_store

            store = get_link_code_store()
            self.assertIsInstance(store, PostgresLinkCodeStore)

    def test_local_mode_returns_local_store_without_attribute_error(self) -> None:
        # The bug this whole module fixes: identity_manager.db_path only
        # exists on the LOCAL IdentityManager. Confirm local mode still
        # resolves it correctly (no AttributeError) via the real singleton.
        with patch("core.storage.factory.settings") as fake_settings:
            fake_settings.is_cloud = False
            from core.account_linking import LinkCodeStore
            from core.storage.factory import get_link_code_store

            store = get_link_code_store()
            self.assertIsInstance(store, LinkCodeStore)


if __name__ == "__main__":
    unittest.main()
