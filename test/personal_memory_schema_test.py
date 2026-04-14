import unittest

from core.personal_memory_schema import (
    parse_markdown_memory,
    serialize_markdown_memory,
    validate_memory_metadata,
)


class PersonalMemorySchemaTests(unittest.TestCase):
    def test_serialize_and_parse_round_trip(self) -> None:
        text = serialize_markdown_memory(
            {
                "topic": "preference",
                "confidence": "confirmed",
                "updated_at": "2026-04-03T10:15:00Z",
            },
            ["Prefers concise replies", "- Avoid filler"],
        )

        parsed = parse_markdown_memory(text)
        self.assertEqual(parsed.metadata["topic"], "preference")
        self.assertEqual(parsed.metadata["confidence"], "confirmed")
        self.assertEqual(
            parsed.lines,
            ["- Prefers concise replies", "- Avoid filler"],
        )

    def test_validate_metadata_rejects_unknown_topic(self) -> None:
        with self.assertRaises(ValueError):
            validate_memory_metadata({"topic": "unknown"})


if __name__ == "__main__":
    unittest.main()
