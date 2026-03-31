import json
import os
import shutil
import unittest
import uuid
from pathlib import Path

from core.env import load_env


class EnvConfigTests(unittest.TestCase):
    def test_json_config_applies_non_secrets_only(self) -> None:
        temp_dir = Path("tests") / "_tmp" / f"env_config_{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        config_path = temp_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "TURTLE_INTERACTION_MODE": "text",
                    "TURTLE_HISTORY_MAX_TOKENS": 9999,
                    "OPEN_ROUTER_API_KEY_1": "should_not_be_loaded",
                }
            ),
            encoding="utf-8",
        )

        original_config_path = os.environ.get("TURTLE_CONFIG_FILE")
        original_mode = os.environ.get("TURTLE_INTERACTION_MODE")
        original_tokens = os.environ.get("TURTLE_HISTORY_MAX_TOKENS")

        try:
            os.environ["TURTLE_CONFIG_FILE"] = str(config_path)
            load_env(override=True)

            self.assertEqual(os.getenv("TURTLE_INTERACTION_MODE"), "text")
            self.assertEqual(os.getenv("TURTLE_HISTORY_MAX_TOKENS"), "9999")
            self.assertNotEqual(os.getenv("OPEN_ROUTER_API_KEY_1"), "should_not_be_loaded")
        finally:
            if original_config_path is None:
                os.environ.pop("TURTLE_CONFIG_FILE", None)
            else:
                os.environ["TURTLE_CONFIG_FILE"] = original_config_path

            if original_mode is None:
                os.environ.pop("TURTLE_INTERACTION_MODE", None)
            else:
                os.environ["TURTLE_INTERACTION_MODE"] = original_mode

            if original_tokens is None:
                os.environ.pop("TURTLE_HISTORY_MAX_TOKENS", None)
            else:
                os.environ["TURTLE_HISTORY_MAX_TOKENS"] = original_tokens

            if "original_secret" not in locals() or original_secret is None:
                os.environ.pop("OPEN_ROUTER_API_KEY_1", None)
            else:
                os.environ["OPEN_ROUTER_API_KEY_1"] = original_secret

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
