"""
test/readyz_test.py
--------------------
WP0.A: /readyz must prove the cloud backends this deploy needs actually work,
must never hang on a dead backend, and cloud startup must refuse to boot
without DATABASE_URL / REDIS_URL. /healthz must keep reporting the build SHA
for "which commit is this" verification.

Fully offline: probes are monkeypatched, no real Postgres/Redis is touched.
"""
from __future__ import annotations

import asyncio
import importlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps import turtle_server as ts
from core.storage.cloud import CloudBackendUnavailable


class ReadyzLocalModeTest(unittest.TestCase):
    """Local mode (the default, no Postgres/Redis) must not probe anything."""

    def setUp(self) -> None:
        self.client = TestClient(ts.app)

    def test_readyz_local_mode_returns_200_without_probing(self) -> None:
        with patch("apps.turtle_server.settings") as fake_settings:
            fake_settings.is_cloud = False
            with patch("core.storage.cloud.probe_postgres") as pg, patch(
                "core.storage.cloud.probe_redis"
            ) as rd:
                resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 200)
        pg.assert_not_called()
        rd.assert_not_called()


class ReadyzCloudModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(ts.app)

    def test_postgres_probe_fails_returns_503_with_booleans(self) -> None:
        async def fake_pg(timeout=None):
            return False

        async def fake_redis(timeout=None):
            return True

        with patch("apps.turtle_server.settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch("core.storage.cloud.probe_postgres", side_effect=fake_pg), patch(
                "core.storage.cloud.probe_redis", side_effect=fake_redis
            ):
                resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertIs(body["postgres"], False)
        self.assertIs(body["redis"], True)

    def test_both_probes_succeed_returns_200(self) -> None:
        async def fake_ok(timeout=None):
            return True

        with patch("apps.turtle_server.settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch("core.storage.cloud.probe_postgres", side_effect=fake_ok), patch(
                "core.storage.cloud.probe_redis", side_effect=fake_ok
            ):
                resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIs(body["postgres"], True)
        self.assertIs(body["redis"], True)

    def test_hanging_probe_does_not_hang_the_route(self) -> None:
        """A probe that would sleep well past the timeout budget must not make
        /readyz hang — drive this via a shrunk READYZ_TIMEOUT_S rather than
        actually waiting out the real 2s budget, so the test stays fast."""
        from core.storage.cloud import get_pg_pool

        async def hanging_pool():
            await asyncio.sleep(5)
            raise AssertionError("should have timed out long before this")

        async def fake_redis_ok(timeout=None):
            return True

        with patch("apps.turtle_server.settings") as fake_settings:
            fake_settings.is_cloud = True
            with patch("core.storage.cloud.READYZ_TIMEOUT_S", 0.05), patch(
                "core.storage.cloud.get_pg_pool", side_effect=hanging_pool
            ), patch("core.storage.cloud.probe_redis", side_effect=fake_redis_ok):
                resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 503)
        self.assertIs(resp.json()["postgres"], False)


class CloudStartupValidationTest(unittest.TestCase):
    """Cloud startup must refuse to boot without DATABASE_URL / REDIS_URL."""

    def test_missing_database_url_raises_on_startup(self) -> None:
        with patch("apps.turtle_server.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.database_url = None
            fake_settings.redis_url = "redis://fake-upstash.example:6379"
            with self.assertRaises(Exception) as ctx:
                with TestClient(ts.app):
                    pass
        self.assertIn("DATABASE_URL", str(ctx.exception))

    def test_missing_redis_url_raises_on_startup(self) -> None:
        with patch("apps.turtle_server.settings") as fake_settings:
            fake_settings.is_cloud = True
            fake_settings.database_url.get_secret_value.return_value = (
                "postgresql://fake:fake@fake-neon.example/db"
            )
            fake_settings.redis_url = None
            with self.assertRaises(Exception) as ctx:
                with TestClient(ts.app):
                    pass
        self.assertIn("REDIS_URL", str(ctx.exception))

    def test_local_mode_startup_does_not_validate_backends(self) -> None:
        # settings.is_cloud is False by default in the test suite; startup
        # must be a no-op — no DATABASE_URL/REDIS_URL in the environment at all.
        with TestClient(ts.app):
            pass  # must not raise

    def test_blank_database_url_raises_same_as_unset(self) -> None:
        """Regression guard for the exact bug behind commit 71c3378: Vercel once
        had TURTLE_DEPLOY set to an empty string rather than unset. The same
        class of mistake (DATABASE_URL="") must fail startup identically to a
        genuinely unset value — exercised against the REAL settings object (not
        a full mock) so the SecretStr("") falsiness is actually proven, not
        assumed."""
        from core.config import settings as real_settings
        from pydantic import SecretStr
        from unittest.mock import patch as _patch

        with _patch.object(real_settings, "deploy_mode", "cloud"), _patch.object(
            real_settings, "database_url", SecretStr("")
        ), _patch.object(
            real_settings, "redis_url_primary", SecretStr("redis://fake-upstash.example:6379")
        ):
            with self.assertRaises(Exception) as ctx:
                with TestClient(ts.app):
                    pass
        self.assertIn("DATABASE_URL", str(ctx.exception))

    def test_blank_redis_url_raises_same_as_unset(self) -> None:
        """Same regression guard as above, for the REDIS_URL/UPSTASH_REDIS_URL
        pair — redis_url is a computed property, not a plain field, so this
        proves the property's own blank-string handling, not just the guard."""
        from core.config import settings as real_settings
        from pydantic import SecretStr
        from unittest.mock import patch as _patch

        with _patch.object(real_settings, "deploy_mode", "cloud"), _patch.object(
            real_settings, "database_url", SecretStr("postgresql://fake:fake@fake-neon.example/db")
        ), _patch.object(real_settings, "redis_url_primary", SecretStr("")), _patch.object(
            real_settings, "redis_url_upstash", SecretStr("")
        ):
            with self.assertRaises(Exception) as ctx:
                with TestClient(ts.app):
                    pass
        self.assertIn("REDIS_URL", str(ctx.exception))

    def test_whitespace_only_database_url_raises_same_as_unset(self) -> None:
        """bool(SecretStr(" ")) is True, so a whitespace-only DATABASE_URL would
        sail past a plain `not settings.database_url` check unless the startup
        hook strips it first — a stray-space copy-paste is the same class of
        mistake as commit 71c3378's DATABASE_URL="" (just a different
        character), and it must fail startup identically."""
        from core.config import settings as real_settings
        from pydantic import SecretStr
        from unittest.mock import patch as _patch

        with _patch.object(real_settings, "deploy_mode", "cloud"), _patch.object(
            real_settings, "database_url", SecretStr("   ")
        ), _patch.object(
            real_settings, "redis_url_primary", SecretStr("redis://fake-upstash.example:6379")
        ):
            with self.assertRaises(Exception) as ctx:
                with TestClient(ts.app):
                    pass
        self.assertIn("DATABASE_URL", str(ctx.exception))

    def test_whitespace_only_redis_url_raises_same_as_unset(self) -> None:
        """Same regression guard as above for the Redis side. redis_url already
        calls .strip() internally (core/config.py) so this currently passes
        because of that existing behaviour — the test still belongs here to
        pin the symmetry: it will fail if the strip is ever removed."""
        from core.config import settings as real_settings
        from pydantic import SecretStr
        from unittest.mock import patch as _patch

        with _patch.object(real_settings, "deploy_mode", "cloud"), _patch.object(
            real_settings, "database_url", SecretStr("postgresql://fake:fake@fake-neon.example/db")
        ), _patch.object(real_settings, "redis_url_primary", SecretStr("   ")), _patch.object(
            real_settings, "redis_url_upstash", SecretStr("   ")
        ):
            with self.assertRaises(Exception) as ctx:
                with TestClient(ts.app):
                    pass
        self.assertIn("REDIS_URL", str(ctx.exception))


class HealthzShaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(ts.app)

    def test_healthz_reports_build_sha_when_set(self) -> None:
        with patch.dict("os.environ", {"TURTLE_BUILD_SHA": "deadbeef123"}):
            resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["sha"], "deadbeef123")

    def test_healthz_sha_key_present_but_null_when_unset(self) -> None:
        import os as _os

        env = dict(_os.environ)
        env.pop("TURTLE_BUILD_SHA", None)
        with patch.dict("os.environ", env, clear=True):
            resp = self.client.get("/healthz")
        body = resp.json()
        self.assertIn("sha", body)
        self.assertIsNone(body["sha"])


class ReadyzTimeoutBudgetTest(unittest.TestCase):
    """WP0.F: the server-side READYZ_TIMEOUT_S budget, and its env override."""

    def _reload_cloud_module(self):
        import core.storage.cloud as cloud_mod

        importlib.reload(cloud_mod)
        return cloud_mod

    def tearDown(self) -> None:
        # Always reload once more with the env var cleared, so the module's
        # module-level READYZ_TIMEOUT_S doesn't leak a test override into
        # later tests/modules that import core.storage.cloud.
        import os as _os

        _os.environ.pop("TURTLE_READYZ_TIMEOUT_S", None)
        self._reload_cloud_module()

    def test_default_budget_is_eight_seconds_when_unset(self) -> None:
        import os as _os

        _os.environ.pop("TURTLE_READYZ_TIMEOUT_S", None)
        cloud_mod = self._reload_cloud_module()
        self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_valid_override_is_honoured(self) -> None:
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": "3.5"}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 3.5)

    def test_unparseable_override_falls_back_to_default(self) -> None:
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": "abc"}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_empty_override_falls_back_to_default(self) -> None:
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": ""}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_zero_override_falls_back_to_default(self) -> None:
        """0 would make asyncio.wait_for raise TimeoutError instantly, making
        /readyz a permanent 503 in cloud mode — must fall back instead."""
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": "0"}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_negative_override_falls_back_to_default(self) -> None:
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": "-5"}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_nan_override_falls_back_to_default(self) -> None:
        """nan must not slip through a naive `value <= 0` check — every
        comparison against nan is False, so `nan <= 0` is also False. This
        pins that math.isfinite() (not a bare `<= 0` comparison) is what
        catches it: with nan, asyncio.wait_for never times out at all,
        the exact opposite of this module's hang-safety promise."""
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": "nan"}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_infinite_override_falls_back_to_default(self) -> None:
        with patch.dict("os.environ", {"TURTLE_READYZ_TIMEOUT_S": "1e400"}):
            cloud_mod = self._reload_cloud_module()
            self.assertEqual(cloud_mod.READYZ_TIMEOUT_S, 8.0)

    def test_probe_exceeding_raised_budget_still_returns_false(self) -> None:
        """Raising the default to 8.0 must not make the timeout path
        unreachable — pass an explicit small `timeout=` so this stays fast
        rather than actually waiting out 8s."""
        cloud_mod = self._reload_cloud_module()

        async def _run() -> bool:
            async def hanging_pool():
                await asyncio.sleep(5)
                raise AssertionError("should have timed out long before this")

            with patch.object(cloud_mod, "get_pg_pool", side_effect=hanging_pool):
                return await cloud_mod.probe_postgres(timeout=0.05)

        self.assertFalse(asyncio.run(_run()))


if __name__ == "__main__":
    unittest.main()
