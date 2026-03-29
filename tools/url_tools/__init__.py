"""
URL Tools Package

A comprehensive package for URL content extraction and analysis.

This package provides tools for fetching, parsing, and analyzing web content
with advanced features like metadata extraction, content cleaning, and 
structured data representation.

Example Usage:
    # Async usage with existing HTTP client
    from tools.url_tools import fetch_url_content_async, UrlState
    
    async with httpx.AsyncClient() as client:
        result = await fetch_url_content_async(client, "https://example.com")
        print(result.to_formatted_string())
    
    # Synchronous usage
    from tools.url_tools import fetch_url_content_sync
    
    result = fetch_url_content_sync("https://example.com")
    print(result.to_formatted_string())
    
    # For use with Pydantic AI agents
    from tools.url_tools import UrlState
    
    @agent.tool
    async def analyze_url(ctx: RunContext[UrlState], url: str) -> str:
        result = await fetch_url_content_async(ctx.deps.http_client, url)
        return result.to_formatted_string()
"""

__version__ = "1.0.0"
__author__ = "Turtle AI Assistant"
__description__ = "Advanced URL content extraction and analysis tools"

# Import main functionality
from .models import UrlState, UrlAnalysisResult
from .extractor import fetch_url_content_async, fetch_url_content_sync

# Define what gets imported with "from tools.url_tools import *"
__all__ = [
    'UrlState',
    'UrlAnalysisResult', 
    'fetch_url_content_async',
    'fetch_url_content_sync'
]
