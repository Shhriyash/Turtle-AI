"""
test/phase5_cold_start_test.py
------------------------------
WP5-C1 (ledger 5.4): cold-start structure.

The ledger blamed tool-contract reads; profiling showed the cost is
GoogleProvider -> google.genai.Client building SSL contexts, once per
(key x call site) = 9 constructions per AgentManager.rebuild. These tests pin
the structure (not wall-clock, which flakes on CPU speed).
"""
from __future__ import annotations

import os
import threading
import unittest
from unittest import mock

try:
    from apps import turtle_server as ts
    from core import llm_client

    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    _IMPORT_ERROR = _e

_FAKE_KEYS = {
    "GEMINI_API_KEY": "fake-gemini-1",
    "GEMINI_API_KEY_2": "fake-gemini-2",
    "GEMINI_API_KEY_3": "fake-gemini-3",
}


def _gemini_env(**extra: str) -> dict[str, str]:
    env = {k: "" for k in llm_client.GEMINI_KEY_ENV_VARS}
    env["GOOGLE_API_KEY"] = ""
    env.update(extra)
    return env


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class GoogleProviderReuse(unittest.TestCase):
    def test_rebuild_constructs_at_most_one_provider_per_key(self) -> None:
        real = llm_client.GoogleProvider
        calls: list[str] = []

        def counting(*a, **kw):
            calls.append(kw.get("api_key"))
            return real(*a, **kw)

        env = _gemini_env(
            **_FAKE_KEYS,
            MAIN_AGENT_MODEL="gemini:gemini-2.5-pro",
            EMAIL_AGENT_MODEL="gemini:gemini-2.5-pro",
        )
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(llm_client, "GoogleProvider", counting):
            # Fresh cache (if any) so the count is not hidden by an earlier build.
            getattr(llm_client, "_GOOGLE_PROVIDERS", {}).clear()
            ts.agents_mgr.rebuild(ts.config)
        keys = {k for k in calls if k}
        self.assertEqual(keys, set(_FAKE_KEYS.values()))
        self.assertLessEqual(
            len(calls), len(keys),
            f"GoogleProvider constructed {len(calls)}x for {len(keys)} keys: {calls}",
        )

    def test_providers_not_shared_across_keys(self) -> None:
        with mock.patch.dict(os.environ, _gemini_env(**_FAKE_KEYS)):
            a = llm_client.get_google_models("gemini-2.5-flash")
            b = llm_client.get_google_models("gemini-2.5-pro")
        self.assertEqual(len(a), 3)
        self.assertEqual(len({id(m.client) for m in a}), 3)  # distinct per key
        for x, y in zip(a, b):
            self.assertIs(x.client, y.client)  # same key -> shared
            self.assertNotEqual(x.model_name, y.model_name)

    def test_rotated_key_gets_fresh_provider(self) -> None:
        with mock.patch.dict(os.environ, _gemini_env(GEMINI_API_KEY="fake-old")):
            old = llm_client.get_google_models()[0]
        with mock.patch.dict(os.environ, _gemini_env(GEMINI_API_KEY="fake-new")):
            new = llm_client.get_google_models()[0]
        self.assertIsNot(old.client, new.client)


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class ColdStartGuards(unittest.TestCase):
    def test_signal_install_survives_non_main_thread(self) -> None:
        # signal.signal raises ValueError off the main thread.
        err: list[BaseException] = []

        def run() -> None:
            try:
                ts._install_shutdown_handlers()
            except BaseException as e:  # noqa: BLE001
                err.append(e)

        t = threading.Thread(target=run)
        t.start()
        t.join()
        self.assertEqual(err, [])

    def test_ensure_static_dir_tolerates_readonly_fs(self) -> None:
        with mock.patch.object(
            type(ts.STATIC_DIR), "mkdir", side_effect=OSError(30, "Read-only file system")
        ):
            ts._ensure_static_dir()  # must not raise

    def test_httpx_capture_all_off_in_cloud(self) -> None:
        self.assertIs(ts._httpx_capture_all(is_cloud=True), False)
        self.assertIs(ts._httpx_capture_all(is_cloud=False), True)

    def test_tool_contract_cached_per_cloud_flag(self) -> None:
        local = ts._load_tool_contract("recall", False)
        cloud = ts._load_tool_contract("recall", True)
        self.assertIn("tasks", local)
        self.assertNotIn("- tasks:", cloud)
        self.assertIs(ts._load_tool_contract("recall", True), cloud)
        self.assertGreaterEqual(ts._read_tool_contract.cache_info().hits, 1)


if __name__ == "__main__":
    unittest.main()
