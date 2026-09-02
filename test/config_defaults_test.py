"""
core/config.py default + field-completeness tests.

Migrated from test_tier2_verification.py (TestG6Config). All assertions target
fields that survive the Phase 4 config trim — no assertion references
personal_memory_dream_pass_enabled (which is being removed).

Covers: settings singleton import, presence of every required API-key field,
the local deploy default, the is_cloud property under TURTLE_DEPLOY=cloud, and
the memory feature-flag defaults.
"""
from __future__ import annotations


class TestG6Config:
    """TurtleSettings covers the required fields with correct defaults."""

    def test_settings_importable(self):
        from core.config import settings, TurtleSettings
        assert settings is not None
        assert TurtleSettings is not None

    def test_required_api_key_fields_present(self):
        from core.config import TurtleSettings
        fields = TurtleSettings.model_fields
        for field in ("openrouter_api_key", "groq_api_key", "tavily_api_key",
                      "deepgram_api_key", "scraped_do_api_key", "auth_secret_key"):
            assert field in fields, f"Missing field: {field}"

    def test_deploy_mode_default_is_local(self):
        import unittest.mock as mock, os
        with mock.patch.dict(os.environ, {}, clear=False):
            from core.config import TurtleSettings
            s = TurtleSettings()
        assert s.deploy_mode == "local"

    def test_is_cloud_property(self):
        import unittest.mock as mock, os
        from core.config import TurtleSettings
        with mock.patch.dict(os.environ, {"TURTLE_DEPLOY": "cloud"}):
            s = TurtleSettings()
        assert s.is_cloud is True

    def test_memory_flags_have_sensible_defaults(self):
        from core.config import TurtleSettings
        s = TurtleSettings()
        assert s.personal_memory_enabled is True
        assert s.personal_memory_max_bytes > 0
