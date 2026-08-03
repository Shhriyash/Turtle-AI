"""Tests for GET /api/memory/profile — the 'what Turtle remembers about you'
viewer that reads the durable, confirmed topic projection from disk."""
import importlib

import pytest
from fastapi.testclient import TestClient

import core.paths as paths
from core.personal_memory_store import PersonalMemoryStore

turtle_server = pytest.importorskip("apps.turtle_server")
app = turtle_server.app


@pytest.fixture
def pm_root(tmp_path, monkeypatch):
    root = tmp_path / "personal"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "PERSONAL_MEMORY_DIR", root, raising=False)
    return root


def _local(monkeypatch):
    # Local mode → _get_user_id_from_request resolves to "local_dev_user".
    monkeypatch.setattr(turtle_server.settings, "deploy_mode", "local", raising=False)


def test_profile_empty_for_unknown_user(pm_root, monkeypatch):
    _local(monkeypatch)
    with TestClient(app) as client:
        r = client.get("/api/memory/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["empty"] is True
    assert body["topics"] == []


def test_profile_returns_stored_topics(pm_root, monkeypatch):
    _local(monkeypatch)
    store = PersonalMemoryStore(user_id="local_dev_user")
    store.write_topic("identity", ["- Name: Test User", "- Dietary Preference: vegan"], {"title": "Identity"})
    store.write_topic("relations", ["- Best Friend: Sam"], {"title": "Relations"})

    with TestClient(app) as client:
        r = client.get("/api/memory/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["empty"] is False
    by_topic = {t["topic"]: t for t in body["topics"]}
    assert "identity" in by_topic
    # Bullet prefix is stripped for display.
    assert "Name: Test User" in by_topic["identity"]["lines"]
    assert "Dietary Preference: vegan" in by_topic["identity"]["lines"]
    assert by_topic["identity"]["title"] == "Identity"
    assert "Best Friend: Sam" in by_topic["relations"]["lines"]


def test_profile_unauthorized_in_cloud(pm_root, monkeypatch):
    monkeypatch.setattr(turtle_server.settings, "deploy_mode", "cloud", raising=False)
    with TestClient(app) as client:
        r = client.get("/api/memory/profile")  # no Bearer, cloud → no local fallback
    assert r.status_code == 401
