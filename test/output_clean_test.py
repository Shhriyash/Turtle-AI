import unittest

from core.output_clean import clean_text_for_model, clean_text_for_tts


class OutputCleanTests(unittest.TestCase):
    def test_clean_text_for_model_strips_markdown(self) -> None:
        text = "**Hello**\n- item one\n[Site](https://example.com)\n```code```"
        cleaned = clean_text_for_model(text)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertIn("Hello", cleaned)
        self.assertIn("item one", cleaned)
        self.assertIn("Site: https://example.com", cleaned)

    def test_clean_text_for_tts_flattens_symbols(self) -> None:
        text = "**Price:** 10/10"
        cleaned = clean_text_for_tts(text)
        self.assertNotIn("*", cleaned)
        self.assertIn("Price.", cleaned)
        self.assertIn("10 out of 10", cleaned)

    def test_clean_text_for_tts_normalizes_common_speech_tokens(self) -> None:
        text = "Mail me @ 25% + tax = final. Link: https://example.com"
        cleaned = clean_text_for_tts(text)
        self.assertIn("at", cleaned)
        self.assertIn("percent", cleaned)
        self.assertIn("plus", cleaned)
        self.assertIn("equals", cleaned)
        self.assertIn("link", cleaned.lower())


if __name__ == "__main__":
    unittest.main()