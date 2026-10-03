"""
test/recall_cloud_task_history_test.py
---------------------------------------
WP2.C (ledger 2.5): task history is a capability that pretended to work in
cloud -- TaskHistoryStore silently no-op'd every write while the `recall`
tool's schema still advertised scope="tasks" as if it worked. This file
proves the honest version:

  1. TaskHistoryStore refuses construction outright in cloud mode (the old
     silent no-op class is gone -- proven by attempting to construct it, not
     just by grepping for dead code).
  2. The `recall` tool's JSON schema handed to the model does NOT offer
     scope="tasks" in cloud, and DOES offer it locally.
  3. The email flow's task-history record is skipped (not attempted, not
     silently swallowed) when task_history_store is None, as it now always
     is in cloud.

Follows test/phase6_admin_test.py's guarded-import pattern so a genuinely
missing optional dep skips rather than erroring collection.
"""
from __future__ import annotations

import asyncio
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

try:
    from pydantic_ai import RunContext

    from apps import turtle_server
    from core.task_history import TaskHistoryStore
    from tools.contracts import EmailArgs

    _IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover - optional deps missing in env
    _IMPORT_ERROR = _e


def _recall_tool_schema(agent) -> dict:
    for tool in agent._function_toolset.tools.values():
        if tool.name == "recall":
            return tool.function_schema.json_schema
    raise AssertionError("recall tool not registered on this agent")


class _FakeSessionStore:
    """Minimal double for core.session_store.SessionStore -- only the surface
    send_email_assistant actually touches."""

    def __init__(self) -> None:
        self.session_id = "sess_cloud_email_test"
        self.message_history: list = []
        self._pending = {
            "recipients": [], "cc_recipients": [], "bcc_recipients": [],
            "subject": "", "content": "", "send_intent": False,
        }

    def get_pending_email(self) -> dict:
        return dict(self._pending)

    async def set_pending_email(self, **kwargs) -> None:
        self._pending.update({k: v for k, v in kwargs.items() if v is not None})

    async def clear_pending_email(self) -> None:
        self._pending = {
            "recipients": [], "cc_recipients": [], "bcc_recipients": [],
            "subject": "", "content": "", "send_intent": False,
        }


class _FakePersonalMemoryStore:
    def load_profile_snapshot(self) -> dict:
        return {}


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class RecallSchemaByDeploymentTest(unittest.TestCase):
    """(2) the recall tool schema varies by deployment."""

    def tearDown(self) -> None:
        # rebuild() mutates the module-global agents_mgr singleton -- always
        # leave it back in local mode so later tests in the same process see
        # the schema they expect.
        with patch.object(turtle_server.settings, "deploy_mode", "local"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)

    def test_cloud_schema_has_no_tasks_scope(self) -> None:
        with patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)
            schema = _recall_tool_schema(turtle_server.agents_mgr.main_assistant)
        scope_enum = schema["properties"]["scope"]["enum"]
        self.assertNotIn("tasks", scope_enum)
        self.assertIn("personal", scope_enum)
        self.assertIn("episodic", scope_enum)
        self.assertIn("working", scope_enum)

    def test_local_schema_has_tasks_scope(self) -> None:
        with patch.object(turtle_server.settings, "deploy_mode", "local"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)
            schema = _recall_tool_schema(turtle_server.agents_mgr.main_assistant)
        scope_enum = schema["properties"]["scope"]["enum"]
        self.assertIn("tasks", scope_enum)


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class RecallContractProseByDeploymentTest(unittest.TestCase):
    """The tool contract prose (a separate signal from the JSON schema --
    see _load_tool_contract in apps/turtle_server.py) must also stop telling
    the model that scope="tasks" is an option in cloud."""

    def tearDown(self) -> None:
        with patch.object(turtle_server.settings, "deploy_mode", "local"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)

    def test_cloud_contract_prose_omits_tasks_scope(self) -> None:
        # _load_tool_contract is a closure local to _register_tools; the
        # public, stable surface is the description actually attached to
        # the registered tool, so read it from there instead of reaching
        # into the closure.
        text = None
        with patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)
            for tool in turtle_server.agents_mgr.main_assistant._function_toolset.tools.values():
                if tool.name == "recall":
                    text = tool.description
                    break
        self.assertIsNotNone(text)
        self.assertNotIn("tasks, working", text)
        with patch.object(turtle_server.settings, "deploy_mode", "local"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)

    def test_local_contract_prose_includes_tasks_scope(self) -> None:
        with patch.object(turtle_server.settings, "deploy_mode", "local"):
            turtle_server.agents_mgr.rebuild(turtle_server.config)
            text = None
            for tool in turtle_server.agents_mgr.main_assistant._function_toolset.tools.values():
                if tool.name == "recall":
                    text = tool.description
                    break
        self.assertIsNotNone(text)
        self.assertIn("tasks, working", text)


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class TaskHistoryStoreCloudConstructionTest(unittest.TestCase):
    """(1) the cloud no-op class is gone -- construction itself must fail."""

    def test_construction_refused_in_cloud(self) -> None:
        from pathlib import Path

        with patch.object(turtle_server.settings, "deploy_mode", "cloud"):
            with self.assertRaises(RuntimeError):
                TaskHistoryStore(Path("test") / "_tmp" / "should_not_exist.jsonl", user_id="usr_x")

    def test_construction_still_works_locally(self) -> None:
        import shutil
        import uuid
        from pathlib import Path

        base = Path("test") / "_tmp" / f"task_history_ctor_{uuid.uuid4().hex}"
        try:
            with patch.object(turtle_server.settings, "deploy_mode", "local"):
                store = TaskHistoryStore(base / "history.jsonl", user_id="usr_x")
            self.assertIsNotNone(store)
        finally:
            shutil.rmtree(base, ignore_errors=True)


@unittest.skipIf(_IMPORT_ERROR is not None, f"app import failed: {_IMPORT_ERROR!r}")
class EmailFlowSkipsTaskRecordInCloudTest(unittest.TestCase):
    """(3) the email flow records no task, and nothing returns a record that
    was not written, when task_history_store is None (as it always is in
    cloud -- see the SharedState construction sites in apps/turtle_server.py)."""

    def _send_email_tool(self):
        for tool in turtle_server.agents_mgr.main_assistant._function_toolset.tools.values():
            if tool.name == "send_email_assistant":
                return tool.function
        raise AssertionError("send_email_assistant tool not registered")

    def test_no_task_history_record_attempted_when_store_is_none(self) -> None:
        state = turtle_server.SharedState(
            http_client=None,
            session_store=_FakeSessionStore(),
            personal_memory_store=_FakePersonalMemoryStore(),
            personal_memory_prompt=None,
            journal_store=None,
            confirmation_gate=None,
            task_history_store=None,  # cloud always constructs None (WP2.C)
            rag_system=None,
            user_id="usr_cloud_email_test",
        )
        ctx = RunContext(deps=state, model=None, usage=None)
        args = EmailArgs(
            query="send an email to test@example.com saying Hello there and subject Hi"
        )

        fake_extraction = AsyncMock(
            return_value=type("R", (), {"output": "{}"})()
        )

        async def _fake_send_with_reservation(idem_key, send_fn):
            return "Email sent successfully! (test)"

        send_tool = self._send_email_tool()

        stdout = io.StringIO()
        with patch.object(turtle_server, "run_agent_with_fallbacks", fake_extraction), \
             patch("tools.idempotency.is_duplicate_invocation", return_value=None), \
             patch("tools.idempotency.send_with_reservation", _fake_send_with_reservation), \
             patch("tools.idempotency.record_invocation", lambda *a, **k: None), \
             redirect_stdout(stdout):
            result = asyncio.run(send_tool(ctx, args))

        self.assertIn("Email sent successfully", result)
        # Old (pre-WP2.C) behaviour unconditionally called
        # ctx.deps.task_history_store.record(...), which on a None store
        # raised AttributeError -- swallowed by the surrounding try/except
        # and logged as "task history record failed". The fix skips the
        # call entirely, so that log line must never appear.
        self.assertNotIn("task history record failed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
