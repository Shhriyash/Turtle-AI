import importlib.util
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


def _load_migration_module():
    script_path = Path("scripts") / "migrate_profile_to_markdown.py"
    spec = importlib.util.spec_from_file_location("migrate_profile_to_markdown", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProfileMigrationTests(unittest.TestCase):
    def test_migration_writes_topic_files_and_index(self) -> None:
        module = _load_migration_module()
        base = Path("test") / "_tmp" / f"profile_migration_{uuid.uuid4().hex}"
        source = base / "profile.json"
        target = base / "personal"
        base.mkdir(parents=True, exist_ok=True)
        try:
            source.write_text(
                json.dumps(
                    {
                        "identity": {
                            "name": "Shriyash",
                            "emails": ["shriyash@example.com", "work@example.com"],
                            "timezone": "Asia/Calcutta",
                        },
                        "preferences": {
                            "response_style": "concise",
                            "humor_level": "low",
                            "email_tone": "formal",
                        },
                        "workflow": {
                            "prefers_draft_before_send": True,
                            "common_recipients": ["team@example.com"],
                            "email_interactions": 3,
                        },
                        "tool_preferences": {"primary_llm": "openrouter"},
                        "meta": {
                            "updated_at": "2026-04-03T10:00:00Z",
                            "version": 1,
                            "workflow_counters": {
                                "common_recipients": {"team@example.com": 3},
                                "email_interactions": 3,
                            },
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = module.migrate_profile_to_markdown(
                profile_path=source,
                target_dir=target,
                force=False,
            )

            self.assertIn("identity", result.written_topics)
            self.assertIn("preferences", result.written_topics)
            self.assertIn("contacts", result.written_topics)
            self.assertTrue((target / "identity.md").exists())
            self.assertTrue((target / "MEMORY.md").exists())
            self.assertIn("Primary email: shriyash@example.com", (target / "identity.md").read_text(encoding="utf-8"))
            self.assertIn("Frequent recipient: team@example.com", (target / "contacts.md").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
