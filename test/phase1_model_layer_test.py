import pytest

from core import health_tracker
from core.llm_client import get_groq_fallback_model, get_groq_model, get_openrouter_models
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.settings import ModelSettings


@pytest.fixture(autouse=True)
def clear_health_tracker_state():
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()
    yield
    with health_tracker._lock:
        health_tracker._cooldown_until.clear()


class _FakeModel:
    model_name = "gemini-2.5-flash"


def _http_error(status_code: int, body=None):
    try:
        return ModelHTTPError(status_code=status_code, model_name="gemini-2.5-flash", body=body)
    except TypeError:
        try:
            return ModelHTTPError(status_code, "gemini-2.5-flash", body)
        except Exception:
            return type("FallbackHTTPError", (Exception,), {"status_code": status_code, "body": body})(
                f"{status_code} {body or ''}"
            )


def _provider_api_key(model):
    provider = getattr(model, "provider", None) or getattr(model, "_provider", None)
    client = getattr(provider, "client", None) or getattr(provider, "_client", None)
    api_key = getattr(client, "api_key", None)
    if api_key is not None:
        return api_key
    return getattr(provider, "api_key", None) or getattr(provider, "_api_key", None)


def _settings_max_tokens(model):
    settings = getattr(model, "settings", None)
    if hasattr(settings, "get"):
        return settings.get("max_tokens")
    return getattr(settings, "max_tokens", None)


def test_transient_cooldown_is_per_rung_not_provider_model_bucket():
    a = _FakeModel()
    b = _FakeModel()

    health_tracker.mark_failure(a, _http_error(429))

    assert health_tracker.is_cooling(a) is True
    assert health_tracker.is_cooling(b) is False


def test_harmony_400_cooldown_is_family_bucket_wide():
    a = _FakeModel()
    b = _FakeModel()

    health_tracker.mark_failure(a, _http_error(400, body="harmony response format error"))

    assert health_tracker.is_cooling(a) is True
    assert health_tracker.is_cooling(b) is True

    health_tracker.mark_success(a)
    health_tracker.mark_success(b)

    assert health_tracker.is_cooling(a) is False
    assert health_tracker.is_cooling(b) is False


def test_402_credits_cool_the_whole_family_not_just_one_key():
    # A 402 (credits exhausted) is an account-level state shared by every key of
    # a provider; cooling one rung for 60s left siblings 402ing every turn.
    a = _FakeModel()
    b = _FakeModel()
    health_tracker.mark_failure(a, _http_error(402, body="requires more credits"))
    assert health_tracker.is_cooling(a) is True
    assert health_tracker.is_cooling(b) is True  # bucket-wide


def test_gemini_tool_pairing_400_cools_the_family():
    # Gemini rejects a tool-turn ordering violation with a 400 that used to get
    # 0s cooldown, so the dead rung was retried every turn (14-36s TTFR live).
    a = _FakeModel()
    b = _FakeModel()
    health_tracker.mark_failure(
        a,
        _http_error(
            400,
            body="Please ensure that function response turn comes immediately after a function call turn.",
        ),
    )
    assert health_tracker.is_cooling(a) is True
    assert health_tracker.is_cooling(b) is True  # bucket-wide


def test_plain_400_still_not_cooled():
    # A generic 400 (genuine bad request) must NOT bench the family.
    a = _FakeModel()
    b = _FakeModel()
    health_tracker.mark_failure(a, _http_error(400, body="malformed field 'foo'"))
    assert health_tracker.is_cooling(a) is False
    assert health_tracker.is_cooling(b) is False


def test_groq_primary_and_fallback_use_distinct_selected_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY2", "k2")

    primary = get_groq_model("llama-3.3-70b-versatile")
    fallback = get_groq_fallback_model("llama-3.1-8b-instant")

    assert _provider_api_key(primary) == "k1"
    assert _provider_api_key(fallback) == "k2"


def test_openrouter_models_raise_max_tokens_floor(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY_1", "x")

    models = get_openrouter_models("google/gemini-2.5-flash", settings=ModelSettings(max_tokens=1024))

    assert models
    assert all((_settings_max_tokens(model) or 0) >= 2048 for model in models)
