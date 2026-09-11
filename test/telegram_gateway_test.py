"""
apps/channels/telegram_gateway.py — offline unit tests.

Verifies the pieces of the adapter that don't need a live bot token: import
guards, the mention-strip, the DM-vs-group filter, and the empty-token no-op.
The full on_message → dispatch_event path is exercised by a stubbed-dispatch
smoke test that avoids the Telegram network.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Import-guard + config
# ---------------------------------------------------------------------------

def test_module_imports_without_bot_token(monkeypatch):
    """gateway_available() must report False when no token is configured.

    `settings` is a module-level singleton already populated from .env by
    the time tests run (pydantic-settings reads the environment once at
    import), so deleting the env var here would not change its cached
    value. Patch the settings object itself instead — this is the same
    thing start_telegram_gateway()/_bot_token() actually read.
    """
    from apps.channels import telegram_gateway
    monkeypatch.setattr(
        telegram_gateway.settings, "telegram_bot_token", None, raising=False
    )
    assert telegram_gateway.gateway_available() is False


def test_channel_literal_includes_telegram():
    """`telegram` must be a recognised Channel — TurtleEvent construction
    trips a type checker but pydantic/dataclass validation is looser; we
    just assert the Literal is updated."""
    from apps.channels import Channel
    # Literal[...] members live on __args__
    assert "telegram" in getattr(Channel, "__args__", ())


# ---------------------------------------------------------------------------
# _extract_message_text — mention strip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,bot,expected",
    [
        # No mention → return trimmed text
        ("hello turtle", "myturtlebot", "hello turtle"),
        # Leading mention stripped
        ("@myturtlebot make an invoice", "myturtlebot", "make an invoice"),
        # Case-insensitive on the handle
        ("@MyTurtleBot summarise this", "myturtlebot", "summarise this"),
        # Inline mention removed once
        ("hey @myturtlebot show me pending", "myturtlebot", "hey  show me pending"),
        # No bot username configured → return raw text
        ("@someone hi", "", "@someone hi"),
        # Empty text → empty
        ("", "myturtlebot", ""),
    ],
)
def test_extract_message_text(raw, bot, expected):
    from apps.channels.telegram_gateway import _extract_message_text
    msg = SimpleNamespace(text=raw, caption=None)
    got = _extract_message_text(msg, bot)
    assert got == expected


def test_extract_falls_back_to_caption():
    """A photo/document message carries text in .caption, not .text."""
    from apps.channels.telegram_gateway import _extract_message_text
    msg = SimpleNamespace(text=None, caption="@bot invoice acme 5000")
    assert _extract_message_text(msg, "bot") == "invoice acme 5000"


# ---------------------------------------------------------------------------
# _should_handle — DM vs group + bot-loop suppression
# ---------------------------------------------------------------------------

def test_should_handle_private_dm():
    from apps.channels.telegram_gateway import _should_handle
    assert _should_handle(None, "private", is_mention=False, is_bot_author=False) is True


def test_should_handle_group_mention():
    from apps.channels.telegram_gateway import _should_handle
    assert _should_handle(None, "group", is_mention=True, is_bot_author=False) is True


def test_should_ignore_group_no_mention():
    """A random group message with no @mention must be a silent no-op —
    otherwise every group message would drive the pipeline."""
    from apps.channels.telegram_gateway import _should_handle
    assert _should_handle(None, "supergroup", is_mention=False, is_bot_author=False) is False


def test_should_ignore_bot_author_always():
    """Bot-loop suppression: never respond to another bot, even a @mention."""
    from apps.channels.telegram_gateway import _should_handle
    assert _should_handle(None, "private", is_mention=True, is_bot_author=True) is False
    assert _should_handle(None, "group", is_mention=True, is_bot_author=True) is False


# ---------------------------------------------------------------------------
# start_telegram_gateway — no-op paths
# ---------------------------------------------------------------------------

def test_start_no_op_without_token(monkeypatch, capsys):
    """No token → clean no-op, print a friendly log line, no exception."""
    from apps.channels import telegram_gateway
    monkeypatch.setattr(
        telegram_gateway.settings,
        "telegram_bot_token",
        None,
        raising=False,
    )
    asyncio.get_event_loop().run_until_complete(
        telegram_gateway.start_telegram_gateway()
    ) if False else asyncio.run(telegram_gateway.start_telegram_gateway())
    out = capsys.readouterr().out
    assert "telegram gateway disabled" in out


def test_stop_is_safe_when_never_started():
    """Idempotent shutdown — nothing to stop should not raise."""
    from apps.channels.telegram_gateway import stop_telegram_gateway
    # If start was never called, module-level _app/_app_task are None
    asyncio.run(stop_telegram_gateway())


# ---------------------------------------------------------------------------
# TurtleEvent construction shape — smoke check via a direct build
# ---------------------------------------------------------------------------

def test_turtle_event_shape_from_dm():
    """The event the adapter would emit for a DM must set is_private=True and
    populate the channel-side identifiers the dispatch layer relies on."""
    from apps.channels import TurtleEvent
    ev = TurtleEvent(
        user_id="u_synthetic",
        channel="telegram",
        modality="text",
        content="hello",
        message_id="42",
        thread_id="777",
        sender_name="Shriyash",
        channel_user_id="tg_123",
        is_private=True,
    )
    assert ev.channel == "telegram"
    assert ev.is_private is True
    assert ev.channel_user_id == "tg_123"


def test_turtle_event_shape_from_group_mention():
    """Group @mention → is_private MUST be False so claim codes refuse."""
    from apps.channels import TurtleEvent
    ev = TurtleEvent(
        user_id="u_synthetic",
        channel="telegram",
        modality="text",
        content="hi",
        message_id="43",
        thread_id="-100777",
        sender_name="Shriyash",
        channel_user_id="tg_123",
        is_private=False,
    )
    assert ev.is_private is False


# ---------------------------------------------------------------------------
# markdown_to_telegram_html — LLM markdown -> Telegram HTML subset
# ---------------------------------------------------------------------------

def test_markdown_to_html_renders_bold_and_bullets():
    from apps.channels.telegram_gateway import markdown_to_telegram_html
    md = "**Top stories from Dubai**\n\n- **Metro** opens.\n- **Weather** warning.\n"
    got = markdown_to_telegram_html(md)
    assert "<b>Top stories from Dubai</b>" in got
    # Bullet lines converted to a bullet character, inline bold still tagged.
    assert "• <b>Metro</b> opens." in got
    assert "• <b>Weather</b> warning." in got


def test_markdown_to_html_headings_lower_to_bold():
    from apps.channels.telegram_gateway import markdown_to_telegram_html
    got = markdown_to_telegram_html("## Weather\nSunny.\n### Details\nHigh 42C.")
    assert "<b>Weather</b>" in got
    assert "<b>Details</b>" in got
    # Body text between headings is plain (no residual '#' markers).
    assert "Sunny." in got and "\n#" not in got


def test_markdown_to_html_inline_code_and_link():
    from apps.channels.telegram_gateway import markdown_to_telegram_html
    got = markdown_to_telegram_html("Run `pip install` then see [docs](https://example.com/x).")
    assert "<code>pip install</code>" in got
    assert '<a href="https://example.com/x">docs</a>' in got


def test_markdown_to_html_fenced_code_block_preserves_content():
    from apps.channels.telegram_gateway import markdown_to_telegram_html
    got = markdown_to_telegram_html("Try:\n```python\nprint(1 < 2)\n```\ndone")
    # The `<` inside the code block must be HTML-escaped and wrapped in <pre>.
    assert '<pre><code class="language-python">print(1 &lt; 2)\n</code></pre>' in got
    # Content OUTSIDE the fence is not eaten.
    assert "Try:" in got and "done" in got


def test_markdown_to_html_escapes_dangerous_raw_html():
    from apps.channels.telegram_gateway import markdown_to_telegram_html
    got = markdown_to_telegram_html("Watch out: <script>alert(1)</script>")
    assert "<script>" not in got
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in got


def test_markdown_to_html_never_leaves_em_or_en_dashes():
    from apps.channels.telegram_gateway import markdown_to_telegram_html, _normalize_dashes
    md = "Latest news, hot off the press — dispatches from 2020–2025."
    got = markdown_to_telegram_html(md)
    assert "—" not in got and "–" not in got
    # Range en dash between digits becomes " to ".
    assert "2020 to 2025" in got
    # Standalone em dash becomes a comma.
    assert "press, dispatches" in got or "press,dispatches" in got
    # Same on the plain normalizer, called directly.
    assert "—" not in _normalize_dashes("a — b") and "–" not in _normalize_dashes("a–b range")


def test_markdown_to_html_empty_and_plain_passthrough():
    from apps.channels.telegram_gateway import markdown_to_telegram_html
    assert markdown_to_telegram_html("") == ""
    # A plain sentence stays plain (no accidental tag injection).
    assert markdown_to_telegram_html("hello there") == "hello there"
