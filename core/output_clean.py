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
    """Strip markdown and punctuation patterns that sound bad in speech."""
    cleaned = clean_text_for_model(text)
    cleaned = re.sub(r"\b([A-Z])\s*\.\s*([A-Z])\b", r"\1 \2", cleaned)
    cleaned = cleaned.replace(":", ". ")
    cleaned = cleaned.replace("/", " or ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
