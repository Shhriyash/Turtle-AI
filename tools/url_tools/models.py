"""
Data models and types for URL tools package.
"""

from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class UrlState:
    """State for URL operations containing HTTP client"""
    http_client: httpx.AsyncClient


@dataclass
class UrlAnalysisResult:
    """Result of URL content analysis"""
    title: str
    description: Optional[str]
    keywords: Optional[str]
    headings: list[str]
    content: str
    links: list[str]
    url: str
    success: bool
    error_message: Optional[str] = None

    def to_formatted_string(self) -> str:
        """Convert analysis result to formatted string for display"""
        if not self.success:
            return f"Error: {self.error_message}"

        result_parts = [f"Title: {self.title}"]

        if self.description:
            result_parts.append(f"Description: {self.description}")

        if self.keywords:
            result_parts.append(f"Keywords: {self.keywords}")

        if self.headings:
            result_parts.append("Page Structure:\n" + "\n".join(self.headings[:10]))

        result_parts.append(f"Main Content:\n{self.content}")

        if self.links:
            shown_links = self.links[:10]
            result_parts.append("Key Links:\n" + "\n".join(shown_links))
            if len(self.links) > 10:
                result_parts.append(f"Additional Links: {len(self.links) - 10} more not shown")

        result_parts.append(f"Source: {self.url}")

        return "\n\n".join(result_parts)
