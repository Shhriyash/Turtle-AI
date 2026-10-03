from __future__ import annotations

import re

# URLs and bare domains, so TTS never reads a link aloud. Covers schemed URLs
# (https://…), www.… hosts, and bare domains with a known TLD and optional path
# (timesofindia.com, reuters.com/world). The TLD list keeps ordinary words with a
# dot (e.g. "index.html", "3.5") from being swallowed.
_URL_OR_DOMAIN_RE = re.compile(
    r"""(?ix)
    \b(
        https?://\S+
      | www\.\S+
      | (?:[a-z0-9-]+\.)+
        (?:com|org|net|edu|gov|mil|io|co|ai|in|dev|news|info|me|app|xyz|uk|us|ca|au|de|fr|jp|cn|gg|tv|fm)
        (?:/\S*)?
    )
    """
)


def clean_text_for_model(text: str, *, flatten_links: bool = True) -> str:
    """Normalize tool output before passing it back to the main agent.

    ``flatten_links`` rewrites ``[text](url)`` to ``text: url`` — right for text
    the MODEL reads, but wrong for text the CHAT renders (the UI turns real
    markdown links into clean clickable anchors). Display callers pass
    ``flatten_links=False`` (see ``clean_text_for_display``).
    """
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if flatten_links:
        cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1: \2", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"^[ \t]*[#>*-]+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_text_for_display(text: str) -> str:
    """Clean a reply for the CHAT, preserving markdown links so the UI renders
    them as clickable anchors (``formatMessage`` in web/js/chat.js)."""
    return clean_text_for_model(text, flatten_links=False)


# ---------------------------------------------------------------------------
# Untrusted tool-result envelope (WP1.H / ledger 1b.2)
# ---------------------------------------------------------------------------
# Tool results (a fetched page, a search snippet, a place review) are
# attacker-influenced text handed to the model with nothing marking it as
# data. These helpers sanitise a tool's raw string return and wrap it in
# ``<untrusted source="...">...</untrusted>`` so the model — per the single
# system-prompt rule that references this tag — treats the contents as data,
# never instructions. Applied once, at the tool-registration loop in
# apps/turtle_server.py, so every registered tool is covered regardless of
# what shape its closure returns internally (see the WP1.H recon notes for
# why ToolResult.to_agent_string() is NOT sufficient by itself).
#
# Kept separate from clean_text_for_model rather than folded into it: that
# function already has several callers outside the tool-result path (email
# drafts, calendar confirmations) and changing its behaviour would ripple
# into all of them. A dedicated function applied only at the envelope
# boundary keeps this control's blast radius to exactly the tools it's
# meant to cover.

# C0 control characters, excluding tab/newline/CR which are ordinary
# whitespace in tool output worth keeping.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# A tool result containing the literal closing delimiter could terminate the
# envelope early and make everything after it read as trusted text — this is
# exactly the attack the envelope exists to stop, so it must be neutralised
# unconditionally (case-insensitively, allowing incidental whitespace inside
# the tag, since the model would recognise any of those variants as "close").
_UNTRUSTED_CLOSE_RE = re.compile(r"</\s*untrusted\s*>", re.IGNORECASE)

# Bare tool-result URLs, collected for the "done" frame so the client can
# allow-list them (see WP1.H item 5). Deliberately scheme-anchored (unlike
# core.output_clean's TTS domain-matcher) — only a real https?:// URL a tool
# actually returned is worth surfacing to the client as clickable.
_ENVELOPE_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+[^\s<>\"')\],.;:!?]")


def sanitize_for_envelope(text: str, *, max_chars: int) -> str:
    """Sanitise a raw tool result before it is wrapped as untrusted data.

    Order matters: strip control characters and well-formed-markdown-link
    syntax, neutralise a literal closing delimiter, THEN truncate. Truncating
    first could slice a neutralised-but-still-long closing sequence back into
    something dangerous; truncating last guarantees the envelope's own
    closing tag (appended by ``wrap_untrusted`` after this returns) is never
    split, because nothing after this function's return touches the tag.
    """
    if not text:
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", text)
    # "](" turns well-formed markdown into a link when the model's own
    # citation format re-parses it later; stripping it here deliberately
    # breaks any markdown embedded in tool output. That's fine — this text
    # is DATA the model reads, not text rendered to the user, so losing
    # markdown formatting inside the envelope costs nothing.
    cleaned = cleaned.replace("](", "] (")
    cleaned = _UNTRUSTED_CLOSE_RE.sub("</ untrusted>", cleaned)
    if len(cleaned) > max_chars:
        cleaned = (
            f"{cleaned[:max_chars]}\n\n"
            "[Output truncated: tool result was too long.]"
        )
    return cleaned


def wrap_untrusted(source: str, text: str) -> str:
    """Wrap already-sanitised tool text in the untrusted-source envelope.

    ``source`` is always the tool's registry name (never a URL or other
    attacker-influenced value) so the attribute itself can't carry injected
    content. Applied even to empty-string results and to error/invalid/
    rate_limited results: an upstream error message (e.g. a fetch failure
    that echoes back part of the requested page) can carry attacker text
    just as easily as a success payload, and an unwrapped empty result would
    make "was this tool's output enveloped" a property that depends on what
    the tool happened to return rather than a constant guarantee the model
    can rely on.
    """
    return f'<untrusted source="{source}">{text}</untrusted>'


def extract_tool_result_urls(text: str) -> list[str]:
    """Pull https?:// URLs out of a (post-sanitisation) tool result.

    Used to populate the "done" frame's tool-sourced URL list — the client
    renders an anchor only for a URL that appears in that list, labelled
    with the host, never for a URL the model merely wrote in prose.
    """
    if not text:
        return []
    return _ENVELOPE_URL_RE.findall(text)


def clean_text_for_tts(text: str) -> str:
    """Render model text into speech-friendly plain language."""
    cleaned = clean_text_for_model(text)

    # Expand common written forms so TTS sounds natural.
    cleaned = re.sub(r"\b(\d+)\s*/\s*(\d+)\b", r"\1 out of \2", cleaned)
    cleaned = re.sub(r"\b([A-Z])\s*\.\s*([A-Z])\b", r"\1 \2", cleaned)
    # Replace every URL / bare domain with the word "link" so it is never spoken.
    cleaned = _URL_OR_DOMAIN_RE.sub("link", cleaned)

    replacements = {
        "&": " and ",
        "@": " at ",
        "%": " percent ",
        "=": " equals ",
        "+": " plus ",
        "*": " ",
        "#": " number ",
        "/": " or ",
    }
    for token, replacement in replacements.items():
        cleaned = cleaned.replace(token, replacement)

    # Flatten punctuation that creates awkward pauses in speech.
    cleaned = cleaned.replace(";", ". ")
    cleaned = cleaned.replace(":", ". ")
    cleaned = re.sub(r"[\[\]{}<>|_^~]", " ", cleaned)
    cleaned = re.sub(r"\.{3,}", ".", cleaned)
    cleaned = re.sub(r"[!?]{2,}", "!", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()