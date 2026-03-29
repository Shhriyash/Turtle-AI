import unittest

from core.web_search import SearchResult, format_search_results


class WebSearchFormatTests(unittest.TestCase):
    def test_format_search_results_plain_text(self) -> None:
        results = [
            SearchResult(
                title="Laptop 1",
                url="https://example.com/laptop-1",
                snippet="High performance gaming laptop",
            ),
            SearchResult(
                title="Laptop 2",
                url="https://example.com/laptop-2",
                snippet="Another strong option",
            ),
        ]

        formatted = format_search_results("gaming laptops amazon.in", results)

        self.assertIn("Web results for query: gaming laptops amazon.in", formatted)
        self.assertIn("1. Laptop 1", formatted)
        self.assertIn("Snippet: High performance gaming laptop", formatted)
        self.assertIn("URL: https://example.com/laptop-1", formatted)
        self.assertNotIn("*", formatted)

    def test_format_search_results_handles_empty(self) -> None:
        formatted = format_search_results("unknown query", [])
        self.assertIn("No web results found", formatted)


if __name__ == "__main__":
    unittest.main()
