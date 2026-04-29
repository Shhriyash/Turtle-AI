from __future__ import annotations

import re


def clean_text_for_model(text: str) -> str:
    """Normalize tool output before passing it back to the main agent."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1: \2", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"^[ \t]*[#>*-]+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_text_for_tts(text: str) -> str:
    """Render model text into speech-friendly plain language."""
    cleaned = clean_text_for_model(text)

    # Expand common written forms so TTS sounds natural.
    cleaned = re.sub(r"\b(\d+)\s*/\s*(\d+)\b", r"\1 out of \2", cleaned)
    cleaned = re.sub(r"\b([A-Z])\s*\.\s*([A-Z])\b", r"\1 \2", cleaned)
    cleaned = re.sub(r"https?://\S+", "link", cleaned)

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