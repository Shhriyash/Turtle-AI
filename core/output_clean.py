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