"""
test/cron_liveness_test.py
---------------------------
Ledger 5.8: cron-tick liveness is surfaced on /readyz (informational) and on
the admin page (red past CRON_TICK_STALE_AFTER_S).

The load-bearing property: a stale tick must NEVER change /readyz's status
code. .github/workflows/deploy-vercel.yml promotes to production only on
HTTP 200 from /readyz, and the measured tick gap is hours, so coupling the two
would turn every deploy into an outage.

Offline: cron_state_store.get_last_tick_at is patched; no Postgres is touched.
Postgres failures raise the real psycopg exception types.
"""
from __future__ import annotations

import re
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import psycopg
import psycopg_pool
from fastapi.testclient import TestClient
from pydantic import SecretStr

from apps import admin_routes
from apps import turtle_server as ts

_GET = "core.storage.cloud.cron_state_store.get_last_tick_at"
_ADMIN_HTML = Path(__file__).resolve().parent.parent / "web" / "admin.html"


async def _ok(timeout=None):
    return True


async def _down(timeout=None):
    return False


def _ago(seconds: float) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


class ReadyzCronFieldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(ts.app)

    def _readyz(self, last_tick, pg=_ok, redis=_ok):
        getter = (
            patch(_GET, side_effect=last_tick)
            if isinstance(last_tick, BaseException)
            else patch(_GET, return_value=last_tick)
        )
        with patch("apps.turtle_server.settings") as fs, patch(
            "apps.admin_routes.settings"
        ) as fs2:
            fs.is_cloud = True
            fs2.is_cloud = True
            with patch("core.storage.cloud.probe_postgres", side_effect=pg), patch(
                "core.storage.cloud.probe_redis", side_effect=redis
            ), getter:
                return self.client.get("/readyz")

    def test_never_ticked_is_null_not_zero(self) -> None:
        resp = self._readyz(None)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("cron_last_tick_age_s", body)
        self.assertIsNone(body["cron_last_tick_age_s"])
        self.assertEqual(body["cron_tick_status"], "never")

    def test_recent_tick_reports_age(self) -> None:
        resp = self._readyz(_ago(120))
        body = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(body["cron_last_tick_age_s"], 120, delta=5)
        self.assertEqual(body["cron_tick_status"], "ok")

    def test_stale_tick_reports_stale(self) -> None:
        resp = self._readyz(_ago(admin_routes.CRON_TICK_STALE_AFTER_S + 600))
        body = resp.json()
        self.assertGreater(body["cron_last_tick_age_s"], admin_routes.CRON_TICK_STALE_AFTER_S)
        self.assertEqual(body["cron_tick_status"], "stale")

    def test_stale_tick_does_not_change_status_code(self) -> None:
        """DEPLOY-GATE GUARD: deploy-vercel.yml promotes only on HTTP 200. A
        tick days old (the 60-day public-repo disable case) must still be 200
        when both backends are healthy."""
        for age in (admin_routes.CRON_TICK_STALE_AFTER_S + 1, 86400, 90 * 86400):
            resp = self._readyz(_ago(age))
            self.assertEqual(resp.status_code, 200, f"age={age}")
            self.assertEqual(resp.json()["cron_tick_status"], "stale")
        self.assertEqual(self._readyz(None).status_code, 200)

    def test_unhealthy_backend_still_503_with_fresh_tick(self) -> None:
        resp = self._readyz(_ago(5), pg=_down)
        self.assertEqual(resp.status_code, 503)
        self.assertIs(resp.json()["postgres"], False)

    def test_postgres_error_during_read_fails_soft(self) -> None:
        for exc in (
            psycopg.OperationalError("connection refused"),
            psycopg_pool.PoolTimeout("couldn't get a connection after 30s"),
            psycopg.errors.UndefinedTable('relation "cron_state" does not exist'),
        ):
            resp = self._readyz(exc)
            self.assertEqual(resp.status_code, 200, type(exc).__name__)
            body = resp.json()
            self.assertIsNone(body["cron_last_tick_age_s"])
            self.assertEqual(body["cron_tick_status"], "unknown")
            self.assertIs(body["postgres"], True)

    def test_read_error_does_not_mask_a_real_503(self) -> None:
        resp = self._readyz(psycopg.OperationalError("down"), pg=_down)
        self.assertEqual(resp.status_code, 503)

    def test_tz_naive_last_tick_does_not_500_the_deploy_gate(self) -> None:
        """cron_liveness() runs inside /readyz's asyncio.gather, so ANYTHING it
        raises becomes a 500 -- and deploy-vercel.yml promotes only on 200, so
        that would block every deploy, not just degrade a field.

        cron_state.last_tick_at is TIMESTAMPTZ and psycopg returns it tz-aware,
        so a naive value should be impossible. This pins the fail-soft anyway,
        because the cost of being wrong is "no deploy can ever promote" and the
        subtraction was originally outside the try/except that guards the read.
        """
        naive = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=120)
        resp = self._readyz(naive)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body["cron_last_tick_age_s"])
        self.assertEqual(body["cron_tick_status"], "unknown")
        self.assertIs(body["postgres"], True)

    def test_local_mode_has_no_cron_field(self) -> None:
        with patch("apps.turtle_server.settings") as fs:
            fs.is_cloud = False
            with patch(_GET) as getter:
                resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("cron_last_tick_age_s", resp.json())
        getter.assert_not_called()


class AdminCronContractTest(unittest.TestCase):
    """Server side AND consumer side of the admin cron field."""

    def setUp(self) -> None:
        self.client = TestClient(ts.app)

    def _get(self, last_tick=None, token="s3cret", cloud=True, side_effect=None):
        hdr = {"X-Admin-Token": token} if token else {}
        getter = (
            patch(_GET, side_effect=side_effect)
            if side_effect is not None
            else patch(_GET, return_value=last_tick)
        )
        with patch.object(
            admin_routes.settings, "admin_token", SecretStr("s3cret")
        ), patch.object(
            admin_routes.settings, "deploy_mode", "cloud" if cloud else "local"
        ), patch.object(
            admin_routes.identity_manager, "init_db", return_value=None
        ), patch.object(
            admin_routes.identity_manager, "list_users", return_value=[]
        ), getter as g:
            resp = self.client.get("/admin/users", headers=hdr)
        return resp, g

    def test_admin_users_carries_cron_block(self) -> None:
        resp, _ = self._get(_ago(30 * 60 + 120))
        self.assertEqual(resp.status_code, 200, resp.text)
        cron = resp.json()["cron"]
        self.assertEqual(cron["stale_after_s"], admin_routes.CRON_TICK_STALE_AFTER_S)
        self.assertEqual(cron["status"], "stale")
        self.assertGreater(cron["last_tick_age_s"], 1800)

    def test_just_under_threshold_is_ok(self) -> None:
        resp, _ = self._get(_ago(30 * 60 - 60))
        self.assertEqual(resp.json()["cron"]["status"], "ok")

    def test_admin_users_never_ticked(self) -> None:
        resp, _ = self._get(None)
        cron = resp.json()["cron"]
        self.assertIsNone(cron["last_tick_age_s"])
        self.assertEqual(cron["status"], "never")

    def test_cron_state_requires_admin_token(self) -> None:
        resp, g = self._get(_ago(5), token=None)
        self.assertEqual(resp.status_code, 401)
        g.assert_not_called()
        resp, g = self._get(_ago(5), token="wrong")
        self.assertEqual(resp.status_code, 401)
        g.assert_not_called()

    def test_local_mode_reports_not_applicable(self) -> None:
        resp, g = self._get(None, cloud=False)
        self.assertEqual(resp.json()["cron"]["status"], "not_applicable")
        g.assert_not_called()

    def test_postgres_error_in_admin_fails_soft(self) -> None:
        resp, _ = self._get(side_effect=psycopg.OperationalError("down"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["cron"]["status"], "unknown")

    def test_page_reads_the_fields_the_server_sends(self) -> None:
        """Consumer side: admin.html must reference every key the server emits
        and must take the threshold from the payload, not a literal."""
        html = _ADMIN_HTML.read_text(encoding="utf-8")
        for key in ("cron", "last_tick_age_s", "stale_after_s", "status"):
            self.assertIn(key, html, f"admin.html never reads cron key {key!r}")
        self.assertIn('id="cron-tick"', html)
        self.assertIn("cron-red", html)
        self.assertIsNone(re.search(r"\b1800\b", html), "threshold must come from the server")
        for status in ("stale", "never", "unknown", "not_applicable"):
            self.assertIn(status, html)

    def test_admin_page_served_has_cron_element(self) -> None:
        resp = self.client.get("/admin")
        self.assertEqual(resp.status_code, 200)
        self.assertIn('id="cron-tick"', resp.text)


if __name__ == "__main__":
    unittest.main()
