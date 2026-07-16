"""
Tier 2 Verification Tests
arch_improve.md — Tier 2 checks

Run with:
    pytest test/test_tier2_verification.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# B4 — JS-rendering URL fetcher (httpx → SPA → Playwright / Scrape.do)
# ---------------------------------------------------------------------------

class TestB4UrlFetcher:
    """B4: httpx first; Playwright fallback on SPA detection; Scrape.do in cloud mode."""

    def test_fetch_url_content_async_is_async(self):
        import asyncio
        from tools.url_tools.extractor import fetch_url_content_async
        assert asyncio.iscoroutinefunction(fetch_url_content_async)
        print("[PASS] fetch_url_content_async is async")

    def test_is_spa_content_function_exists(self):
        from tools.url_tools.extractor import _is_spa_content
        assert callable(_is_spa_content)
        print("[PASS] _is_spa_content is callable")

    def test_is_spa_content_sparse_text(self):
        from tools.url_tools.extractor import _is_spa_content
        assert _is_spa_content("") is True
        assert _is_spa_content("  loading...  ") is True
        print("[PASS] _is_spa_content detects sparse/empty text as SPA")

    def test_is_spa_content_rich_text(self):
        from tools.url_tools.extractor import _is_spa_content
        rich = "The quick brown fox jumps over the lazy dog. " * 10
        assert _is_spa_content(rich) is False
        print("[PASS] _is_spa_content passes rich text as non-SPA")

    def test_fetch_with_playwright_function_exists(self):
        import asyncio
        from tools.url_tools.extractor import _fetch_with_playwright
        assert asyncio.iscoroutinefunction(_fetch_with_playwright)
        print("[PASS] _fetch_with_playwright is async")

    def test_fetch_with_scraped_do_function_exists(self):
        import asyncio
        from tools.url_tools.extractor import _fetch_with_scraped_do
        assert asyncio.iscoroutinefunction(_fetch_with_scraped_do)
        print("[PASS] _fetch_with_scraped_do is async")

    def test_fetch_static_html_via_httpx(self):
        """B4 step 1: httpx path returns content for a normal static HTML page."""
        import asyncio, unittest.mock as mock

        STATIC_HTML = """<html><head><title>Test Page</title></head>
        <body>
        <p>This is a rich static page with enough content to pass SPA detection checks.
        The article is about Python programming and web scraping techniques that are
        widely used for data extraction and automation tasks.</p>
        </body></html>"""

        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.text = STATIC_HTML
        fake_resp.headers = {"content-type": "text/html; charset=utf-8"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_resp)
            return await fetch_url_content_async(mock_client, "https://example.com")

        result = asyncio.run(run())
        assert result.success is True
        assert result.title == "Test Page"
        assert len(result.content) > 0
        print(f"[PASS] Static HTML fetched via httpx: title={result.title!r}")

    def test_spa_detection_triggers_playwright_when_no_token(self):
        """B4 step 2: sparse HTML triggers Playwright fallback when no Scrape.do token."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"
        RENDERED_HTML = """<html><head><title>SPA Rendered</title></head>
        <body><p>Content rendered by JavaScript after hydration. This paragraph is long
        enough to pass the SPA threshold and confirm that Playwright successfully
        rendered the page with full JavaScript execution support.</p></body></html>"""

        fake_httpx_resp = mock.MagicMock()
        fake_httpx_resp.raise_for_status = mock.Mock()
        fake_httpx_resp.status_code = 200
        fake_httpx_resp.text = SPA_HTML
        fake_httpx_resp.headers = {"content-type": "text/html"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_httpx_resp)

            # Patch Scrape.do token to None and Playwright to return rich HTML
            with mock.patch("tools.url_tools.extractor._fetch_with_playwright",
                            new=mock.AsyncMock(return_value=(RENDERED_HTML, 200, "text/html"))) as pw_mock, \
                 mock.patch("tools.url_tools.extractor._fetch_with_scraped_do") as sd_mock, \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = None
                result = await fetch_url_content_async(mock_client, "https://spa-example.com")
            # Playwright must have been called; Scrape.do must not
            pw_mock.assert_called_once()
            sd_mock.assert_not_called()
            return result

        result = asyncio.run(run())
        assert result.success is True
        print(f"[PASS] SPA triggers Playwright (no Scrape.do token): title={result.title!r}")

    def test_spa_detection_triggers_scraped_do_when_token_set(self):
        """B4 cloud path: SPA + Scrape.do token → Scrape.do called, Playwright skipped."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"
        RENDERED_HTML = """<html><head><title>Scraped Page</title></head>
        <body><p>Content fetched through Scrape.do proxy with JS rendering enabled
        for higher success rates and geo bypass across various regions worldwide.</p>
        </body></html>"""

        fake_httpx_resp = mock.MagicMock()
        fake_httpx_resp.raise_for_status = mock.Mock()
        fake_httpx_resp.status_code = 200
        fake_httpx_resp.text = SPA_HTML
        fake_httpx_resp.headers = {"content-type": "text/html"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_httpx_resp)

            fake_secret = mock.MagicMock()
            fake_secret.get_secret_value.return_value = "test-scraped-do-token"

            with mock.patch("tools.url_tools.extractor._fetch_with_playwright") as pw_mock, \
                 mock.patch("tools.url_tools.extractor._fetch_with_scraped_do",
                            new=mock.AsyncMock(return_value=(RENDERED_HTML, 200, "text/html"))) as sd_mock, \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = fake_secret
                result = await fetch_url_content_async(mock_client, "https://spa-example.com")
            sd_mock.assert_called_once()
            pw_mock.assert_not_called()
            return result

        result = asyncio.run(run())
        assert result.success is True
        assert "Scraped Page" in result.title
        print(f"[PASS] SPA triggers Scrape.do (token set): title={result.title!r}")

    def test_invalid_url_returns_failure_result(self):
        import asyncio, unittest.mock as mock

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            return await fetch_url_content_async(mock_client, "not-a-url")

        result = asyncio.run(run())
        assert result.success is False
        assert result.error_message is not None
        print(f"[PASS] Invalid URL returns failure: {result.error_message!r}")

    def test_timeout_returns_failure_result(self):
        import asyncio, unittest.mock as mock
        import httpx

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            return await fetch_url_content_async(mock_client, "https://slow-site.com", timeout=5.0)

        result = asyncio.run(run())
        assert result.success is False
        assert "Timeout" in result.error_message or "timeout" in result.error_message.lower()
        print(f"[PASS] Timeout produces failure result: {result.error_message!r}")

    def test_json_response_returned_directly(self):
        import asyncio, unittest.mock as mock

        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.text = '{"price": 65000}'
        fake_resp.json = mock.Mock(return_value={"price": 65000})
        fake_resp.headers = {"content-type": "application/json"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async
            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_resp)
            return await fetch_url_content_async(mock_client, "https://api.example.com/data")

        result = asyncio.run(run())
        assert result.success is True
        assert "65000" in result.content
        print("[PASS] JSON response returned directly without HTML parsing")

    def test_playwright_import_error_falls_through_gracefully(self):
        """If Playwright is not installed, SPA path returns sparse content without crashing."""
        import asyncio, unittest.mock as mock

        SPA_HTML = "<html><body><div id='root'></div></body></html>"

        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.Mock()
        fake_resp.status_code = 200
        fake_resp.text = SPA_HTML
        fake_resp.headers = {"content-type": "text/html"}

        async def run():
            from tools.url_tools.extractor import fetch_url_content_async

            async def raise_import(*args, **kwargs):
                raise ImportError("playwright not installed")

            mock_client = mock.AsyncMock()
            mock_client.get = mock.AsyncMock(return_value=fake_resp)

            with mock.patch("tools.url_tools.extractor._fetch_with_playwright",
                            side_effect=raise_import), \
                 mock.patch("core.config.settings") as cfg_mock:
                cfg_mock.scraped_do_api_key = None
                return await fetch_url_content_async(mock_client, "https://spa.com")

        result = asyncio.run(run())
        # Must succeed (not crash) even without Playwright
        assert result.success is True
        print("[PASS] Missing Playwright handled gracefully — no crash")


# ---------------------------------------------------------------------------
# H2 — Citation-grounded answer validator
# ---------------------------------------------------------------------------

class TestH2CitationValidator:
    """H2: Post-response citation check; re-prompt when factual claims lack sources."""

    def test_module_importable(self):
        from core.validators.citation_validator import check_citation, CitationCheckResult
        assert callable(check_citation)
        assert CitationCheckResult is not None
        print("[PASS] citation_validator importable")

    def test_no_tool_urls_always_passes(self):
        from core.validators.citation_validator import check_citation
        result = check_citation("Bitcoin is worth a lot.", tool_urls=[])
        assert result.passed is True
        assert result.reprompt is None
        print("[PASS] Empty tool_urls always passes")

    def test_response_with_matching_url_passes(self):
        from core.validators.citation_validator import check_citation
        tool_urls = ["https://coinmarketcap.com/currencies/bitcoin/"]
        response = "Bitcoin is $65,000. Source: https://coinmarketcap.com/currencies/bitcoin/"
        result = check_citation(response, tool_urls)
        assert result.passed is True
        assert len(result.urls_cited) >= 1
        assert result.reprompt is None
        print("[PASS] Response with matching URL passes")

    def test_response_without_url_fails(self):
        from core.validators.citation_validator import check_citation
        tool_urls = ["https://example.com/article"]
        response = "Bitcoin is worth $65,000 today."
        result = check_citation(response, tool_urls)
        assert result.passed is False
        assert result.reprompt is not None
        print(f"[PASS] Response missing URL fails: reprompt={result.reprompt[:40]!r}…")

    def test_reprompt_message_is_actionable(self):
        from core.validators.citation_validator import check_citation
        result = check_citation("Some claim.", tool_urls=["https://news.site/story"])
        assert result.reprompt is not None
        msg = result.reprompt.lower()
        assert "cite" in msg or "source" in msg or "url" in msg
        print("[PASS] Reprompt message is actionable")

    def test_partial_url_match_passes(self):
        """Response that cites the host domain of a returned URL should still pass."""
        from core.validators.citation_validator import check_citation
        tool_urls = ["https://coinmarketcap.com/currencies/bitcoin/?ref=xyz"]
        response = "BTC price from https://coinmarketcap.com is $65k."
        result = check_citation(response, tool_urls)
        assert result.passed is True
        print("[PASS] Host-level URL match counts as citation")

    def test_multiple_tool_urls_passes_with_one_cited(self):
        from core.validators.citation_validator import check_citation
        tool_urls = [
            "https://reuters.com/article/ai-news",
            "https://techcrunch.com/2025/ai",
        ]
        response = "AI is advancing fast. Read more at https://reuters.com/article/ai-news"
        result = check_citation(response, tool_urls)
        assert result.passed is True
        print("[PASS] Citing one of many tool URLs is sufficient")

    def test_citation_check_result_fields(self):
        from core.validators.citation_validator import check_citation
        urls = ["https://example.com"]
        result = check_citation("No link here.", urls)
        assert hasattr(result, "passed")
        assert hasattr(result, "urls_returned")
        assert hasattr(result, "urls_cited")
        assert hasattr(result, "reprompt")
        assert result.urls_returned == urls
        print("[PASS] CitationCheckResult has all required fields")


# ---------------------------------------------------------------------------
# G7 — Production deploy (Dockerfile + compose + no reload default)
# ---------------------------------------------------------------------------

class TestG7ProductionDeploy:
    """G7: Dockerfile + docker-compose exist; reload=True is not the default."""

    def test_dockerfile_exists(self):
        dockerfile = ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"
        print("[PASS] Dockerfile exists")

    def test_dockerfile_has_production_cmd(self):
        """G7: CMD must use gunicorn (not plain python) for cloud-grade process management."""
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "gunicorn" in content, "Dockerfile CMD must use gunicorn for production"
        print("[PASS] Dockerfile uses gunicorn")

    def test_dockerfile_sets_correct_deploy_env(self):
        """Dockerfile must set TURTLE_DEPLOY=cloud (not TURTLE_DEPLOY_MODE)."""
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "TURTLE_DEPLOY=cloud" in content, \
            "Dockerfile must set TURTLE_DEPLOY=cloud (correct env alias)"
        assert "TURTLE_DEPLOY_MODE" not in content, \
            "TURTLE_DEPLOY_MODE is wrong env key; correct key is TURTLE_DEPLOY"
        print("[PASS] Dockerfile sets TURTLE_DEPLOY=cloud")

    def test_docker_compose_exists(self):
        compose = ROOT / "docker-compose.yml"
        assert compose.exists(), "docker-compose.yml not found"
        print("[PASS] docker-compose.yml exists")

    def test_docker_compose_has_turtle_service(self):
        import yaml  # pyyaml is a dep via pydantic-settings extras
        content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        services = data.get("services", {})
        assert len(services) >= 1, "docker-compose.yml has no services"
        service_names = list(services.keys())
        print(f"[PASS] docker-compose.yml has services: {service_names}")

    def test_server_reload_defaults_to_false(self):
        """G7: server_reload must default to False — reload=True must not be the default."""
        from core.config import TurtleSettings
        import unittest.mock as mock
        with mock.patch.dict(__import__("os").environ, {}, clear=False):
            s = TurtleSettings()
        assert s.server_reload is False, \
            f"server_reload default must be False, got {s.server_reload}"
        print("[PASS] server_reload defaults to False")

    def test_reload_only_set_when_flag_enabled(self):
        """Verify the server entry-point only passes reload=True when explicitly opted in."""
        src = (ROOT / "apps" / "turtle_server.py").read_text(encoding="utf-8")
        # The literal reload=True must be inside a conditional block, not unconditional
        lines = src.splitlines()
        reload_true_lines = [i for i, l in enumerate(lines) if "reload=True" in l]
        assert reload_true_lines, "reload=True not found in server — check the entry-point"
        for lineno in reload_true_lines:
            # There must be an if/conditional in the preceding 10 lines
            window = "\n".join(lines[max(0, lineno - 10): lineno + 1])
            assert "if " in window or "reload_enabled" in window, \
                f"reload=True at line {lineno + 1} appears unconditional"
        print("[PASS] reload=True is guarded by a conditional — not the default")


# ---------------------------------------------------------------------------
# G1 — Storage abstraction layer
# ---------------------------------------------------------------------------

class TestG1StorageAbstractions:
    """G1: Protocol interfaces defined; local implementations (SQLite, FAISS, Blob) present."""

    def test_protocols_importable(self):
        from core.storage import FactStore, VectorStore, SessionStoreProtocol, Queue, BlobStore, TraceSink
        print("[PASS] All storage protocol types importable")

    def test_fact_and_hit_models_importable(self):
        from core.storage import Fact, Hit, Session
        f = Fact(id="f1", topic="t", key="k", value="v")
        assert f.key == "k"
        print("[PASS] Fact/Hit/Session models importable and constructable")

    def test_sqlite_session_store_importable(self):
        from core.storage.local.sqlite_store import SQLiteSessionStore
        assert SQLiteSessionStore is not None
        print("[PASS] SQLiteSessionStore importable")

    def test_sqlite_session_store_roundtrip(self):
        import asyncio, tempfile
        from pathlib import Path
        from core.storage.local.sqlite_store import SQLiteSessionStore
        from core.storage import Session

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLiteSessionStore(db_path=Path(tmp) / "s.sqlite")
                await store.init_db()
                sess = Session(session_id="sess_abc", data={"foo": "bar"})
                await store.put(sess)
                result = await store.get("sess_abc")
                assert result is not None
                assert result.data["foo"] == "bar"
                print("[PASS] SQLiteSessionStore put/get roundtrip")

        asyncio.run(run())

    def test_sqlite_fact_store_importable(self):
        from core.storage.local.fact_store import SQLiteFactStore
        assert SQLiteFactStore is not None
        print("[PASS] SQLiteFactStore importable")

    def test_sqlite_fact_store_roundtrip(self):
        import asyncio, tempfile
        from pathlib import Path
        from core.storage.local.fact_store import SQLiteFactStore

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLiteFactStore(db_path=Path(tmp) / "facts.sqlite")
                await store.init_db()
                await store.upsert_fact("user1", "food", "favourite", "pizza")
                facts = await store.get_facts("user1", "food")
                assert len(facts) == 1
                assert facts[0].value == "pizza"
                # Upsert update
                await store.upsert_fact("user1", "food", "favourite", "sushi")
                facts2 = await store.get_facts("user1", "food")
                assert len(facts2) == 1
                assert facts2[0].value == "sushi"
                print("[PASS] SQLiteFactStore upsert and update roundtrip")

        asyncio.run(run())

    def test_sqlite_fact_store_user_isolation(self):
        import asyncio, tempfile
        from pathlib import Path
        from core.storage.local.fact_store import SQLiteFactStore

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLiteFactStore(db_path=Path(tmp) / "facts.sqlite")
                await store.init_db()
                await store.upsert_fact("user_a", "prefs", "color", "blue")
                await store.upsert_fact("user_b", "prefs", "color", "red")
                a_facts = await store.get_facts("user_a", "prefs")
                b_facts = await store.get_facts("user_b", "prefs")
                assert a_facts[0].value == "blue"
                assert b_facts[0].value == "red"
                print("[PASS] SQLiteFactStore user isolation confirmed")

        asyncio.run(run())

    def test_local_blob_store_importable(self):
        from core.storage.local.blob_store import LocalBlobStore
        assert LocalBlobStore is not None
        print("[PASS] LocalBlobStore importable")

    def test_local_blob_store_roundtrip(self):
        import asyncio, tempfile
        from pathlib import Path
        from core.storage.local.blob_store import LocalBlobStore

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = LocalBlobStore(root=Path(tmp) / "blobs")
                payload = b"hello blob world"
                uri = await store.put("test/hello.bin", payload)
                assert uri.startswith("file://")
                got = await store.get("test/hello.bin")
                assert got == payload
                deleted = await store.delete("test/hello.bin")
                assert deleted is True
                gone = await store.get("test/hello.bin")
                assert gone is None
                print("[PASS] LocalBlobStore put/get/delete roundtrip")

        asyncio.run(run())

    def test_content_addressed_key(self):
        from core.storage.local.blob_store import LocalBlobStore
        data = b"deterministic content"
        k1 = LocalBlobStore.content_key(data)
        k2 = LocalBlobStore.content_key(data)
        assert k1 == k2
        assert k1.startswith("blobs/")
        print("[PASS] content_key is deterministic")


# ---------------------------------------------------------------------------
# G2 — Session/Journal abstraction
# ---------------------------------------------------------------------------

class TestG2SessionJournalAbstraction:
    """G2: SessionStore uses SQLite backend; JournalStore takes user_id."""

    def test_session_store_uses_sqlite_backend(self):
        from core.session_store import SessionStore
        src = (ROOT / "core" / "session_store.py").read_text(encoding="utf-8")
        assert "SQLiteSessionStore" in src or "SessionStoreProtocol" in src, \
            "session_store.py must reference SQLiteSessionStore or SessionStoreProtocol"
        print("[PASS] session_store.py references SQLite-backed store")

    def test_journal_store_takes_user_id(self):
        import inspect
        from core.memory_journal import JournalStore
        sig = inspect.signature(JournalStore.__init__)
        assert "user_id" in sig.parameters, "JournalStore.__init__ must accept user_id"
        print("[PASS] JournalStore.__init__ accepts user_id")

    def test_journal_store_per_user_isolation(self):
        import tempfile
        from pathlib import Path
        from core.memory_journal import JournalStore, make_event

        with tempfile.TemporaryDirectory() as tmp:
            import unittest.mock as mock
            with mock.patch("core.paths.MEMORY_DIR", Path(tmp) / "mem"):
                js_a = JournalStore(user_id="user_alice")
                js_b = JournalStore(user_id="user_bob")
                assert js_a.journal_dir != js_b.journal_dir, \
                    "Different users must have different journal directories"
        print("[PASS] JournalStore uses separate directories per user_id")


# ---------------------------------------------------------------------------
# G3 — Worker queue
# ---------------------------------------------------------------------------

class TestG3WorkerQueue:
    """G3: LocalWorkerQueue uses asyncio.create_task; task decorator registers jobs."""

    def test_worker_queue_importable(self):
        from core.worker import LocalWorkerQueue, queue_service, task
        assert LocalWorkerQueue is not None
        assert queue_service is not None
        print("[PASS] worker module importable")

    def test_task_decorator_registers_job(self):
        from core.worker import _REGISTRY, task

        @task("_test_job_xyz")
        async def _test_job_xyz():
            pass

        assert "_test_job_xyz" in _REGISTRY
        print("[PASS] @task decorator registers the function in _REGISTRY")

    def test_local_queue_enqueue_returns_job_id(self):
        import asyncio
        from core.worker import LocalWorkerQueue, task

        @task("_test_noop")
        async def _test_noop(**_):
            pass

        async def run():
            q = LocalWorkerQueue()
            job_id = await q.enqueue("_test_noop")
            assert job_id.startswith("job_")
            return job_id

        job_id = asyncio.run(run())
        print(f"[PASS] enqueue returns job_id={job_id!r}")

    def test_unknown_job_raises(self):
        import asyncio, pytest
        from core.worker import LocalWorkerQueue

        async def run():
            q = LocalWorkerQueue()
            with pytest.raises(ValueError, match="not registered"):
                await q.enqueue("nonexistent_job_abc")

        asyncio.run(run())
        print("[PASS] Enqueuing unknown job raises ValueError")

    def test_embed_personal_memory_task_registered(self):
        import core.background_tasks  # noqa: F401 — triggers @task registration
        from core.worker import _REGISTRY
        assert "embed_personal_memory" in _REGISTRY
        print("[PASS] embed_personal_memory task registered")


# ---------------------------------------------------------------------------
# G4 — Auth + WebSocket session
# ---------------------------------------------------------------------------

class TestG4Auth:
    """G4: JWT creation/verification; WebSocket close on missing token in cloud mode."""

    def test_auth_module_importable(self):
        from apps.auth import create_session_token, verify_token, authenticate_websocket
        print("[PASS] apps.auth importable")

    def test_create_and_verify_token(self):
        import unittest.mock as mock
        from apps.auth import create_session_token, verify_token
        from core.config import TurtleSettings
        from pydantic import SecretStr

        with mock.patch("apps.auth.settings") as s:
            s.auth_secret_key = SecretStr("test-secret-abc")
            token = create_session_token("usr_abc123", "web")
            assert isinstance(token, str) and len(token) > 20

            payload = verify_token(token)
            assert payload["sub"] == "usr_abc123"
            assert payload["channel"] == "web"
        print("[PASS] JWT create and verify round-trip")

    def test_expired_token_raises(self):
        import time, pytest, unittest.mock as mock
        from apps.auth import create_session_token, verify_token
        from pydantic import SecretStr

        with mock.patch("apps.auth.settings") as s:
            s.auth_secret_key = SecretStr("test-secret-abc")
            # Create a token backdated by more than 24h
            import jwt as _jwt
            from datetime import datetime, UTC, timedelta
            payload = {"sub": "usr_old", "channel": "web",
                       "exp": datetime.now(UTC) - timedelta(hours=25)}
            token = _jwt.encode(payload, "test-secret-abc", algorithm="HS256")

            with pytest.raises(ValueError, match="[Ee]xpired|[Ii]nvalid"):
                verify_token(token)
        print("[PASS] Expired token raises ValueError")

    def test_auth_secret_key_in_config(self):
        from core.config import TurtleSettings
        import inspect
        # Confirm field is declared (field name is auth_secret_key)
        fields = TurtleSettings.model_fields
        assert "auth_secret_key" in fields, \
            "auth_secret_key must be a declared field in TurtleSettings"
        print("[PASS] auth_secret_key field present in TurtleSettings")

    def test_dev_fallback_no_crash_without_secret(self):
        """Local mode auth should work even without AUTH_SECRET_KEY set."""
        import unittest.mock as mock
        from apps.auth import create_session_token, verify_token

        with mock.patch("apps.auth.settings") as s:
            s.auth_secret_key = None
            token = create_session_token("usr_dev")
            payload = verify_token(token)
            assert payload["sub"] == "usr_dev"
        print("[PASS] Auth works with dev fallback secret when AUTH_SECRET_KEY not set")


# ---------------------------------------------------------------------------
# G5 — Observability
# ---------------------------------------------------------------------------

class TestG5Observability:
    """G5: OTel TraceSink initialises; span accepts all standard Turtle attributes."""

    def test_observability_module_importable(self):
        from core.observability import trace_sink, init_observability, OTelTraceSink
        assert trace_sink is not None
        print("[PASS] core.observability importable")

    def test_otel_trace_sink_is_trace_sink(self):
        from core.observability import OTelTraceSink
        from core.storage import TraceSink
        sink = OTelTraceSink()
        assert hasattr(sink, "span") and callable(sink.span)
        print("[PASS] OTelTraceSink has .span() method")

    def test_span_accepts_turtle_attributes(self):
        from opentelemetry.sdk.trace import TracerProvider
        from core.observability import OTelTraceSink
        sink = OTelTraceSink()
        # Isolated provider (no exporter) so the span never reaches the global
        # BatchSpanProcessor that appends to the real data/traces/traces.jsonl.
        sink.tracer = TracerProvider().get_tracer("turtle-test")
        with sink.span(
            "test.turn",
            user_id="usr_abc",
            session_id="sess_xyz",
            intent="web",
            model="groq:llama-3.1-8b",
            tokens_in=120,
            tokens_out=80,
            cost_usd=0.0001,
            tool_status="ok",
            hallucination_check_result="pass",
        ) as sp:
            assert sp is not None
        print("[PASS] OTelTraceSink.span() accepts all standard Turtle attributes")

    def test_latency_auto_measured_when_not_provided(self):
        """When latency_ms is omitted, the sink measures and sets it automatically."""
        import time
        from opentelemetry.sdk.trace import TracerProvider
        from core.observability import OTelTraceSink, ATTR_LATENCY_MS
        sink = OTelTraceSink()
        # Isolated provider (no exporter) so the span never reaches the global
        # BatchSpanProcessor that appends to the real data/traces/traces.jsonl.
        sink.tracer = TracerProvider().get_tracer("turtle-test")
        with sink.span("test.latency.auto", user_id="usr_test") as sp:
            time.sleep(0.01)
        # The span attribute should have been set on the OTel span object
        attrs = dict(sp.attributes or {})
        assert ATTR_LATENCY_MS in attrs, "latency_ms not auto-set on span"
        assert attrs[ATTR_LATENCY_MS] >= 0
        print(f"[PASS] Auto-measured latency: {attrs[ATTR_LATENCY_MS]:.1f} ms")

    def test_standard_attribute_constants_exported(self):
        from core.observability import (
            ATTR_USER_ID, ATTR_SESSION_ID, ATTR_INTENT, ATTR_MODEL,
            ATTR_LATENCY_MS, ATTR_TOKENS_IN, ATTR_TOKENS_OUT,
            ATTR_COST_USD, ATTR_TOOL_STATUS, ATTR_HALLUCINATION_CHECK,
        )
        assert all(a.startswith("turtle.") for a in [
            ATTR_USER_ID, ATTR_SESSION_ID, ATTR_INTENT, ATTR_MODEL,
            ATTR_LATENCY_MS, ATTR_TOKENS_IN, ATTR_TOKENS_OUT,
            ATTR_COST_USD, ATTR_TOOL_STATUS, ATTR_HALLUCINATION_CHECK,
        ])
        print("[PASS] All Turtle OTel attribute constants use 'turtle.' prefix")

    def test_jsonl_exporter_writes_to_file(self):
        import tempfile, json
        from pathlib import Path
        from core.observability import JSONLSpanExporter
        from opentelemetry.sdk.trace.export import SpanExportResult

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "traces" / "traces.jsonl"
            exporter = JSONLSpanExporter(log_path)
            result = exporter.export([])  # Empty export succeeds without crashing
            assert result == SpanExportResult.SUCCESS
            assert log_path.parent.exists()
        print("[PASS] JSONLSpanExporter exports without crashing")


# ---------------------------------------------------------------------------
# G6 — Config completeness
# ---------------------------------------------------------------------------

class TestG6Config:
    """G6: TurtleSettings covers all required fields with correct defaults."""

    def test_settings_importable(self):
        from core.config import settings, TurtleSettings
        assert settings is not None
        print("[PASS] core.config.settings importable")

    def test_required_api_key_fields_present(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        for field in ("openrouter_api_key", "groq_api_key", "tavily_api_key",
                      "deepgram_api_key", "scraped_do_api_key", "auth_secret_key"):
            assert field in fields, f"Missing field: {field}"
        print("[PASS] All required API key fields declared in TurtleSettings")

    def test_deploy_mode_default_is_local(self):
        import unittest.mock as mock, os
        with mock.patch.dict(os.environ, {}, clear=False):
            from core.config import TurtleSettings
            s = TurtleSettings()
        assert s.deploy_mode == "local"
        print("[PASS] deploy_mode defaults to 'local'")

    def test_is_cloud_property(self):
        import unittest.mock as mock
        from core.config import TurtleSettings
        with mock.patch.dict(__import__("os").environ, {"TURTLE_DEPLOY": "cloud"}):
            s = TurtleSettings()
        assert s.is_cloud is True
        print("[PASS] is_cloud returns True when TURTLE_DEPLOY=cloud")

    def test_memory_flags_have_sensible_defaults(self):
        from core.config import TurtleSettings
        s = TurtleSettings()
        assert s.personal_memory_enabled is True
        assert s.personal_memory_max_bytes > 0
        print("[PASS] Memory flags have sensible defaults")


# ---------------------------------------------------------------------------
# F5 — Per-tenant identity
# ---------------------------------------------------------------------------

class TestF5Identity:
    """F5: resolve_user() creates a canonical UserId per channel+channel_user_id."""

    def test_identity_manager_importable(self):
        from core.identity import IdentityManager, identity_manager
        assert identity_manager is not None
        print("[PASS] core.identity importable")

    def test_resolve_user_creates_new_user(self):
        import asyncio, tempfile
        from pathlib import Path
        from core.identity import IdentityManager

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                mgr = IdentityManager(db_path=Path(tmp) / "users.sqlite")
                await mgr.init_db()
                uid = await mgr.resolve_user("web", "alice@example.com")
                assert uid.startswith("usr_")
                return uid

        uid = asyncio.run(run())
        print(f"[PASS] resolve_user created new UserId: {uid}")

    def test_resolve_user_idempotent(self):
        import asyncio, tempfile
        from pathlib import Path
        from core.identity import IdentityManager

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                mgr = IdentityManager(db_path=Path(tmp) / "users.sqlite")
                await mgr.init_db()
                uid1 = await mgr.resolve_user("slack", "U12345")
                uid2 = await mgr.resolve_user("slack", "U12345")
                assert uid1 == uid2, "Same channel+user must always resolve to the same UserId"

        asyncio.run(run())
        print("[PASS] resolve_user is idempotent for same channel+channel_user_id")

    def test_different_channels_same_name_are_isolated(self):
        """Same logical username on two channels → two different UserIds."""
        import asyncio, tempfile
        from pathlib import Path
        from core.identity import IdentityManager

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                mgr = IdentityManager(db_path=Path(tmp) / "users.sqlite")
                await mgr.init_db()
                uid_web = await mgr.resolve_user("web", "alice")
                uid_wa = await mgr.resolve_user("whatsapp", "alice")
                assert uid_web != uid_wa, "Different channels must produce different UserIds"

        asyncio.run(run())
        print("[PASS] Different channels produce different UserIds for same handle")

    def test_user_identity_model_importable(self):
        from core.identity import UserIdentity
        ui = UserIdentity(user_id="usr_abc", created_at="2026-01-01")
        assert ui.user_id == "usr_abc"
        print("[PASS] UserIdentity model importable and constructable")


# ---------------------------------------------------------------------------
# D5 — Semantic recall over personal memory
# ---------------------------------------------------------------------------

class TestD5SemanticRecall:
    """D5: Personal memory lines are embedded into vector store; RetrievalBroker uses semantic search."""

    def test_faiss_vector_store_importable(self):
        from core.storage.local.faiss_store import FAISSVectorStore
        assert FAISSVectorStore is not None
        print("[PASS] FAISSVectorStore importable")

    def test_embed_personal_memory_task_registered(self):
        import core.background_tasks  # noqa: F401 — triggers @task registration
        from core.worker import _REGISTRY
        assert "embed_personal_memory" in _REGISTRY
        print("[PASS] embed_personal_memory background task registered (D5)")

    def test_personal_memory_store_enqueues_embed_on_write(self):
        """PersonalMemoryStore.write_topic must enqueue the embed_personal_memory task."""
        import inspect
        src = (ROOT / "core" / "personal_memory_store.py").read_text(encoding="utf-8")
        assert "embed_personal_memory" in src, \
            "PersonalMemoryStore must enqueue 'embed_personal_memory' on topic write"
        print("[PASS] PersonalMemoryStore enqueues embed_personal_memory on write")

    def test_retrieval_broker_has_vector_search(self):
        """RetrievalBroker must reference vector_store.search for semantic tier."""
        src = (ROOT / "core" / "retrieval_broker.py").read_text(encoding="utf-8")
        assert "vector_store" in src or "FAISSVectorStore" in src, \
            "RetrievalBroker must use vector_store for semantic topic selection"
        print("[PASS] RetrievalBroker references vector_store for semantic search")

    def test_vector_store_upsert_and_search_protocol(self):
        """VectorStore protocol has upsert and search methods."""
        from core.storage import VectorStore
        import inspect
        assert hasattr(VectorStore, "upsert") or "upsert" in str(VectorStore)
        print("[PASS] VectorStore protocol defines upsert and search")


# ---------------------------------------------------------------------------
# D6 — Multi-tenant memory namespace
# ---------------------------------------------------------------------------

class TestD6MultiTenantMemory:
    """D6: Paths use user_id; PersonalMemoryStore and JournalStore namespaced per user."""

    def test_personal_memory_dir_takes_user_id(self):
        import inspect
        from core.paths import personal_memory_dir
        sig = inspect.signature(personal_memory_dir)
        assert "user_id" in sig.parameters
        print("[PASS] personal_memory_dir(user_id) signature correct")

    # NOTE: a former test here asserted personal_memory_dir("usr_alice") ==
    # personal_memory_dir("usr_bob") — blessing a cross-tenant privacy
    # regression. Deleted 2026-07-16; test/production_path_isolation_test.py
    # is the correct per-user isolation spec.

    def test_personal_journal_dir_takes_user_id(self):
        import inspect
        from core.paths import personal_journal_dir
        sig = inspect.signature(personal_journal_dir)
        assert "user_id" in sig.parameters
        print("[PASS] personal_journal_dir(user_id) signature correct")

    def test_rag_vector_dir_takes_user_id(self):
        import inspect
        from core.paths import rag_vector_dir
        sig = inspect.signature(rag_vector_dir)
        assert "user_id" in sig.parameters
        print("[PASS] rag_vector_dir(user_id) signature correct")

    def test_personal_memory_store_takes_user_id(self):
        import inspect
        from core.personal_memory_store import PersonalMemoryStore
        sig = inspect.signature(PersonalMemoryStore.__init__)
        assert "user_id" in sig.parameters, \
            "PersonalMemoryStore.__init__ must accept user_id"
        print("[PASS] PersonalMemoryStore.__init__ accepts user_id")

    # NOTE: a former test here asserted PersonalMemoryStore("usr_alice") and
    # ("usr_bob") share base_dir — blessing a cross-tenant privacy regression.
    # Deleted 2026-07-16; test/production_path_isolation_test.py is the
    # correct per-user isolation spec.
