"""
apps/channels/telegram_gateway.py
---------------------------------
F6: Telegram channel adapter — long-polling BOT mode.

This is the "natural conversation" path for Telegram: TURTLE runs as a
registered bot (created via @BotFather → bot token) and replies to direct
messages and @mentions in groups without a slash command. It uses the HTTP Bot
API via python-telegram-bot, which is an OPTIONAL dependency — the import is
guarded so this module (and the whole app / test suite) loads cleanly when the
library is NOT installed. In that case the gateway simply no-ops.

TURTLE runs here strictly as a registered BOT. It is NEVER a user account
(MTProto self-login via api_id / api_hash + phone SMS). Driving a real user
account is technically possible with pyrogram/telethon but sits outside the
adapter contract Discord established: bots only, one identity per platform.

The MTProto app credentials (api_id / api_hash from my.telegram.org) are NOT
used here — the HTTP Bot API only needs the bot token issued by @BotFather.

Required env var:
  TELEGRAM_BOT_TOKEN   Bot token from @BotFather (format "123456:ABC-…").

Bot setup notes:
  - Create the bot via @BotFather → /newbot → copy the token.
  - For the bot to see messages in GROUPS, either mention it (@YourBot ...) or
    disable privacy mode via @BotFather → /setprivacy → Disable. For DM-only
    use, privacy mode can stay ON.
  - is_private is set to True only for the "private" chat type (a DM). A group
    reply is public, and secret-bearing tools (account-link claim codes) will
    refuse to emit into a public chat — matching Discord.
"""
from __future__ import annotations

import asyncio
import re

from core.config import settings

# Guarded import: python-telegram-bot is optional. Absence must be a clean
# no-op, so the app boots and the CI suite passes without it installed.
try:
    from telegram import Update  # type: ignore
    from telegram.ext import (  # type: ignore
        Application,
        ApplicationBuilder,
        MessageHandler,
        filters,
    )
    _TELEGRAM_IMPORT_OK = True
except Exception:  # pragma: no cover - exercised only when the lib is absent
    Update = None  # type: ignore
    Application = None  # type: ignore
    ApplicationBuilder = None  # type: ignore
    MessageHandler = None  # type: ignore
    filters = None  # type: ignore
    _TELEGRAM_IMPORT_OK = False

# Module-level handles so shutdown can reach the running application + task.
_app = None  # type: ignore[var-annotated]
_app_task = None  # type: ignore[var-annotated]

# Telegram's outbound message hard limit is 4096 chars (vs Discord's 2000).
# Leave headroom so trailing markers we may append still fit.
_MAX_REPLY_CHARS = 4000


def gateway_available() -> bool:
    """True when the gateway CAN run: telegram lib importable AND a bot token set."""
    return _TELEGRAM_IMPORT_OK and bool(_bot_token())


def _bot_token() -> str:
    return (
        settings.telegram_bot_token.get_secret_value()
        if settings.telegram_bot_token
        else ""
    )


def _extract_message_text(message, bot_username: str) -> str:
    """Strip a leading @mention of our bot so the pipeline sees clean text.

    Kept separate from the handler so it is unit-testable without a live bot.
    """
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    if not text:
        return ""
    if bot_username:
        # Telegram mentions look like "@BotName" — case-insensitive on the handle.
        needle = f"@{bot_username}"
        low = text.lower()
        low_needle = needle.lower()
        if low.startswith(low_needle):
            text = text[len(needle):]
        else:
            # Mention not at the start — strip a single inline occurrence.
            idx = low.find(low_needle)
            if idx != -1:
                text = text[:idx] + text[idx + len(needle):]
    return text.strip()


def _normalize_dashes(text: str) -> str:
    """Defensive net for the "no em/en dashes" rule: even if a model slips
    one through, the outbound message doesn't. Em dash -> comma+space, en
    dash between digits -> "to" (range), en dash otherwise -> comma+space.
    Kept minimal on purpose so it never turns a real intentional character
    into something surprising.
    """
    if not text:
        return text
    text = re.sub(r"\s*—\s*", ", ", text)
    # An en dash between digits usually means a range ("2020-2025"). Turn it
    # into "to" so the reading stays clear when read aloud or copied.
    text = re.sub(r"(?<=\d)\s*–\s*(?=\d)", " to ", text)
    text = re.sub(r"\s*–\s*", ", ", text)
    return text


# --- Markdown -> Telegram HTML converter -----------------------------------
# Telegram supports a small HTML subset when the message is sent with
# parse_mode="HTML": <b>, <i>, <u>, <s>, <code>, <pre>, <a href>. There is no
# native support for markdown headings, lists, or blockquotes, so we lower
# those to reasonable equivalents (bold headings, bullet characters, plain
# lines with a leading "> "). Anything unrecognised is passed through with
# only the three HTML metacharacters escaped, so a stray "<" in prose can't
# unbalance a tag and reject the whole message.
_HTML_META = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def _html_escape(text: str) -> str:
    return "".join(_HTML_META.get(ch, ch) for ch in text)


# The formatting-inside-line regexes are applied to already-html-escaped
# text; they only look at the LLM's markdown tokens (**, *, _, `, [ ](url)).
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_BOLD_UNDER_RE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_ITALIC_STAR_RE = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)", flags=re.DOTALL)
_ITALIC_UNDER_RE = re.compile(r"(?<![_\w])_(?!\s)(.+?)(?<!\s)_(?!_)", flags=re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", flags=re.DOTALL)
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _apply_inline_markdown(escaped_text: str) -> str:
    """Turn markdown inline tokens in already-HTML-escaped text into tags.

    Order matters: pull inline code first (its contents are literal), then
    the paired-token formats, so a stray ``*`` inside code never accidentally
    starts a bold run.
    """
    # Placeholder-swap for inline code so its content is not touched by the
    # other passes below.
    placeholders: list[str] = []

    def _reserve(html: str) -> str:
        placeholders.append(html)
        return f"\x00PH{len(placeholders) - 1}\x00"

    def _code(m: re.Match[str]) -> str:
        return _reserve(f"<code>{m.group(1)}</code>")

    escaped_text = _INLINE_CODE_RE.sub(_code, escaped_text)

    # Links: [label](url). Both parts are already HTML-escaped, but a raw
    # quote in the URL would break the href attribute, so drop any " inside.
    def _link(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2).replace('"', "%22")
        return _reserve(f'<a href="{url}">{label}</a>')

    escaped_text = _LINK_RE.sub(_link, escaped_text)

    # Bold and italic. The **/__ patterns fire before *//_ so ** doesn't get
    # eaten as two italic runs.
    escaped_text = _BOLD_STAR_RE.sub(r"<b>\1</b>", escaped_text)
    escaped_text = _BOLD_UNDER_RE.sub(r"<b>\1</b>", escaped_text)
    escaped_text = _ITALIC_STAR_RE.sub(r"<i>\1</i>", escaped_text)
    escaped_text = _ITALIC_UNDER_RE.sub(r"<i>\1</i>", escaped_text)
    escaped_text = _STRIKE_RE.sub(r"<s>\1</s>", escaped_text)

    # Restore inline-code / link placeholders.
    def _restore(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return placeholders[idx] if 0 <= idx < len(placeholders) else m.group(0)

    escaped_text = re.sub(r"\x00PH(\d+)\x00", _restore, escaped_text)
    return escaped_text


def markdown_to_telegram_html(text: str) -> str:
    """Convert the model's markdown reply into the Telegram HTML subset.

    Handles: fenced code blocks, headings (`# ` .. `###### `), bullet and
    numbered lists, blockquotes (`> `), inline **bold**, *italic*, `code`,
    ~~strike~~, and `[label](url)` links. Anything else is left as escaped
    text so a stray `<script>` in the reply body renders as text, not tag.
    """
    if not text:
        return ""

    text = _normalize_dashes(text)

    # 1. Fenced code blocks come out first so their contents are never
    #    touched by the inline / block passes below.
    code_blocks: list[str] = []

    def _reserve_code_block(m: re.Match[str]) -> str:
        lang = (m.group(1) or "").strip()
        body = m.group(2)
        escaped_body = _html_escape(body)
        if lang:
            html = f'<pre><code class="language-{_html_escape(lang)}">{escaped_body}</code></pre>'
        else:
            html = f"<pre>{escaped_body}</pre>"
        code_blocks.append(html)
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(
        r"```([A-Za-z0-9_+.\-]*)\n(.*?)```",
        _reserve_code_block,
        text,
        flags=re.DOTALL,
    )

    # 2. Line-oriented block transforms.
    out_lines: list[str] = []
    for raw_line in text.split("\n"):
        # Preserve pure code-block placeholder lines untouched.
        if raw_line.strip().startswith("\x00CB") and raw_line.strip().endswith("\x00"):
            out_lines.append(raw_line)
            continue

        line = raw_line
        # Headings -> bold. Telegram HTML has no <h1..h6>.
        heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading_match:
            content = _apply_inline_markdown(_html_escape(heading_match.group(2)))
            out_lines.append(f"<b>{content}</b>")
            continue

        # Bullet list markers -> bullet character.
        bullet_match = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet_match:
            content = _apply_inline_markdown(_html_escape(bullet_match.group(1)))
            out_lines.append(f"• {content}")
            continue

        # Numbered list markers -> keep the number, keep the text.
        numbered_match = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if numbered_match:
            content = _apply_inline_markdown(_html_escape(numbered_match.group(2)))
            out_lines.append(f"{numbered_match.group(1)}. {content}")
            continue

        # Blockquote -> leading "> " (Telegram has no native blockquote in
        # the HTML subset, but the character reads correctly).
        quote_match = re.match(r"^\s*>\s?(.*)$", line)
        if quote_match:
            content = _apply_inline_markdown(_html_escape(quote_match.group(1)))
            out_lines.append(f"❝ {content}")
            continue

        # Plain line: escape, then apply inline markdown.
        out_lines.append(_apply_inline_markdown(_html_escape(line)))

    html = "\n".join(out_lines)

    # 3. Restore fenced code blocks.
    def _restore_code_block(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return code_blocks[idx] if 0 <= idx < len(code_blocks) else m.group(0)

    html = re.sub(r"\x00CB(\d+)\x00", _restore_code_block, html)
    return html


def _should_handle(message, chat_type: str, is_mention: bool, is_bot_author: bool) -> bool:
    """Filter logic mirroring the Discord gateway's on_message gate.

    Handle when: not a bot author AND (chat is a private DM OR the bot was
    @mentioned in a group). Silent no-op otherwise.
    """
    if is_bot_author:
        return False
    if chat_type == "private":
        return True
    return bool(is_mention)


async def start_telegram_gateway() -> None:
    """Start the Telegram bot as a background task (best-effort).

    Graceful no-op when python-telegram-bot is not installed or no bot token
    is set. Mirrors start_discord_gateway.
    """
    global _app, _app_task
    if not _TELEGRAM_IMPORT_OK or not _bot_token():
        print("LOG: telegram gateway disabled (no python-telegram-bot / no token)", flush=True)
        return
    if _app is not None:
        print("LOG: telegram gateway already running", flush=True)
        return

    # Import locally so type-checkers/readers see the guarded module and the
    # imports happen only when we actually run.
    from apps.channels import TurtleEvent, TurtleResponse, dispatch_event
    from core.identity import identity_manager

    application = ApplicationBuilder().token(_bot_token()).build()

    async def _on_message(update, context) -> None:  # pragma: no cover - live gateway
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        user = message.from_user
        if user is None:
            return

        # Bot-loop suppression: ignore any bot author, including ourselves.
        is_bot_author = bool(getattr(user, "is_bot", False))
        chat_type = getattr(chat, "type", "") or ""

        # Detect an @mention of our own bot (group messages only).
        bot_username = ""
        try:
            bot_username = (context.bot.username or "").lstrip("@")
        except Exception:
            bot_username = ""

        # An @-mention shows up in message.entities as {type: "mention"} whose
        # text matches "@BotName". Cheap contains-check on raw text is enough
        # for the filter; the strip is exact.
        raw_text = getattr(message, "text", "") or getattr(message, "caption", "") or ""
        is_mention = bool(bot_username) and (f"@{bot_username}".lower() in raw_text.lower())

        print(
            f"LOG: telegram on_message from {user.id} chat_type={chat_type} "
            f"is_bot={is_bot_author} mention={is_mention} content_len={len(raw_text)}",
            flush=True,
        )

        if not _should_handle(message, chat_type, is_mention, is_bot_author):
            print("LOG: telegram on_message ignored (not DM, not @mention)", flush=True)
            return

        text = _extract_message_text(message, bot_username)
        if not text:
            print("LOG: telegram on_message dropped — empty text after mention-strip", flush=True)
            return

        try:
            tg_user_id = str(user.id)
            user_id = await identity_manager.resolve_user("telegram", tg_user_id)
            sender_name = (
                getattr(user, "full_name", "")
                or getattr(user, "first_name", "")
                or getattr(user, "username", "")
                or ""
            )
            turtle_event = TurtleEvent(
                user_id=user_id,
                channel="telegram",
                modality="text",
                content=text,
                message_id=str(message.message_id),
                thread_id=str(chat.id),
                sender_name=str(sender_name),
                channel_user_id=tg_user_id,
                # A private chat is a DM (only this user sees the reply). A
                # group @mention is public — do NOT emit claim codes there.
                is_private=(chat_type == "private"),
            )
            print(
                f"LOG: telegram dispatching turn user_id={user_id} text={text[:80]!r}",
                flush=True,
            )
            response: TurtleResponse = await dispatch_event(turtle_event)
            reply_text = (response.content or "…")[:_MAX_REPLY_CHARS]
            # Telegram renders markdown only when we ask it to. Convert the
            # model's markdown to Telegram's small HTML subset and send with
            # parse_mode="HTML"; if the converter produced anything Telegram
            # refuses to parse (e.g. an unbalanced entity), fall back to the
            # plain-text send so the user still gets the message.
            html_reply = markdown_to_telegram_html(reply_text)
            try:
                await message.reply_text(html_reply, parse_mode="HTML")
            except Exception as parse_err:
                print(
                    f"LOG: telegram HTML parse rejected, resending as plain text: {parse_err}",
                    flush=True,
                )
                await message.reply_text(_normalize_dashes(reply_text))
            print(
                f"LOG: telegram replied to {user.id} ({len(response.content or '')} chars)",
                flush=True,
            )
        except Exception as e:
            print(f"LOG: telegram gateway on_message error: {e}", flush=True)

    # Text messages only. Commands (/start etc.) are ignored to keep parity
    # with Discord's "natural conversation" gateway path.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))

    async def _run() -> None:
        # initialize/start/polling all block cooperatively; run them under a
        # detached task so the app startup returns immediately.
        await application.initialize()
        await application.start()
        await application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES if Update is not None else None,
            drop_pending_updates=True,
        )

    _app = application
    _app_task = asyncio.create_task(_run())

    def _log_gateway_exit(task: "asyncio.Task") -> None:
        # Surface a silent connection failure instead of letting the exception
        # die inside the detached task. The common cause is an invalid bot
        # token (Unauthorized).
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(
                f"LOG: telegram gateway stopped: {exc.__class__.__name__}: {exc}",
                flush=True,
            )

    _app_task.add_done_callback(_log_gateway_exit)

    try:
        from core.worker import track_task
        track_task(_app_task)
    except Exception:
        pass

    print("LOG: telegram gateway starting", flush=True)


async def stop_telegram_gateway() -> None:
    """Stop polling, shut the application down, and cancel the runner task."""
    global _app, _app_task
    if _app is not None:
        try:
            if _app.updater is not None and _app.updater.running:
                await _app.updater.stop()
            if _app.running:
                await _app.stop()
                await _app.shutdown()
        except Exception as e:
            print(f"LOG: telegram gateway shutdown error: {e}", flush=True)
    if _app_task is not None:
        try:
            _app_task.cancel()
        except Exception:
            pass
    _app = None
    _app_task = None
