"""
core/streaming_tts.py sentence-splitting behavior tests.

Migrated from test_tier1_verification.py (TestE3StreamingTTS) — the real
behavior assertions only (server-source greps and the TODO-marker source check
are intentionally dropped as test theatre).

Covers: split_into_sentences at boundaries and on empty input, the
SentenceAccumulator streaming buffer (fire-on-boundary + flush remainder), and
the two streaming entry points being async generators.
"""
from __future__ import annotations


class TestE3StreamingTTS:
    """Sentence accumulator splits at boundaries; streaming entry points exist."""

    def test_split_into_sentences_basic(self):
        from core.streaming_tts import split_into_sentences
        sentences = split_into_sentences(
            "Hello world. How are you? I am fine! Let us begin."
        )
        assert len(sentences) >= 3, f"Expected >= 3 sentences, got {sentences}"

    def test_split_handles_empty_string(self):
        from core.streaming_tts import split_into_sentences
        assert split_into_sentences("") == []

    def test_accumulator_fires_on_sentence_boundary(self):
        from core.streaming_tts import SentenceAccumulator
        acc = SentenceAccumulator()
        fired: list[str] = []
        tokens = ["Hello ", "world", ". ", "How ", "are ", "you?"]
        for token in tokens:
            fired.extend(acc.feed(token))
        remainder = acc.flush()
        all_output = fired + remainder
        assert len(all_output) >= 1, "Expected at least one sentence"

    def test_accumulator_flush_returns_remainder(self):
        from core.streaming_tts import SentenceAccumulator
        acc = SentenceAccumulator()
        acc.feed("This is an unfinished thought")
        remainder = acc.flush()
        assert len(remainder) >= 1
        assert "unfinished" in remainder[0]

    def test_stream_tts_from_text_is_async_generator(self):
        import inspect
        from core.streaming_tts import stream_tts_from_text
        assert inspect.isasyncgenfunction(stream_tts_from_text)

    def test_stream_tts_from_token_stream_is_async_generator(self):
        import inspect
        from core.streaming_tts import stream_tts_from_token_stream
        assert inspect.isasyncgenfunction(stream_tts_from_token_stream)
