"""
DDL-and-roundtrip test for core/storage/cloud/calendar_token_store.py
(table: calendar_tokens) against a real Postgres.
"""
from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.cloud


def test_calendar_token_store_put_get_delete_roundtrip() -> None:
    from core.storage.cloud import calendar_token_store as store

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    token_json = '{"refresh_token": "fake-refresh-token-not-real"}'

    assert store.token_exists(user_id) is False
    assert store.get_token_json(user_id) is None

    store.put_token_json(user_id, token_json)
    assert store.token_exists(user_id) is True
    assert store.get_token_json(user_id) == token_json

    updated_json = '{"refresh_token": "fake-refresh-token-updated"}'
    store.put_token_json(user_id, updated_json)
    assert store.get_token_json(user_id) == updated_json

    store.delete_token_json(user_id)
    assert store.token_exists(user_id) is False
    assert store.get_token_json(user_id) is None
