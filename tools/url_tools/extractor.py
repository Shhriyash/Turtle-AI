"""
Core URL content extraction functionality.

This module provides the main URL content fetching and analysis capabilities
that can be used independently of any agent framework.
"""

import re
import json
from urllib.parse import urlparse
from typing import Optional, Union
import httpx
from bs4 import BeautifulSoup

from .models import UrlAnalysisResult


async def fetch_url_content_async(
    http_client: httpx.AsyncClient, 
    url: str, 
    max_content_length: int = 8000,
    timeout: float = 20.0
) -> UrlAnalysisResult:
    """
    Fetch and extract detailed content from a URL with optimized performance.
    
    Args:
        http_client: Async HTTP client for making requests
        url: URL to fetch content from
        max_content_length: Maximum length of content to extract
        timeout: Request timeout in seconds
        
    Returns:
        UrlAnalysisResult containing extracted content and metadata
    """
    
    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return UrlAnalysisResult(
                title="", description=None, keywords=None, headings=[], 
                content="", links=[], url=url, success=False,
                error_message=f"Invalid URL format - {url}"
            )
    except Exception as e:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[], 
            content="", links=[], url=url, success=False,
            error_message=f"URL validation failed - {str(e)}"
        )
    
    try:
        # Enhanced headers for better compatibility
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Optimized request with longer timeout for better content
        response = await http_client.get(
            url, 
            headers=headers,
            timeout=timeout,
            follow_redirects=True
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get content type and encoding
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle different content types
        if 'json' in content_type:
            try:
                json_data = response.json()
                formatted_json = json.dumps(json_data, indent=2)
                return UrlAnalysisResult(
                    title="JSON Content", description="API Response", keywords=None,
                    headings=[], content=formatted_json, links=[], url=url, success=True
                )
            except Exception:
                return UrlAnalysisResult(
                    title="JSON Content", description="API Response", keywords=None,
                    headings=[], content=response.text, links=[], url=url, success=True
                )
        
        elif 'html' not in content_type:
            # For other non-HTML content, return more content
            content = response.text[:2000]
            return UrlAnalysisResult(
                title=f"Content ({content_type})", description=f"Non-HTML content", 
                keywords=None, headings=[], content=content + "...", links=[], 
                url=url, success=True
            )
        
        # Enhanced HTML parsing
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Debug: Check if we actually got HTML content
        html_text = soup.get_text().strip()
        if not html_text:
            # No text content found, possibly a JavaScript-heavy site
            return UrlAnalysisResult(
                title="Dynamic Content Page", 
                description="This page appears to use JavaScript for content loading", 
                keywords=None, headings=[], 
                content="Ã¢Å¡Â Ã¯Â¸Â This webpage appears to load content dynamically using JavaScript. "
                       "The page requires browser rendering to display its full content. "
                       "This could be a single-page application (SPA) or require user authentication.", 
                links=[], url=url, success=True
            )
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '').strip() if meta_desc else None
        
        # Extract keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords = meta_keywords.get('content', '').strip() if meta_keywords else None
        
        # Remove unwanted elements more comprehensively
        for element in soup(["script", "style", "nav", "footer", "header", "aside", 
                            "noscript", "iframe", "embed", "object", "form", "button"]):
            element.decompose()
        
        # Try to extract main content areas first
        main_content = None
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content', 
            '.post-content', '.entry-content', '.article-content', '#content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Use main content if found, otherwise use body
        content_source = main_content if main_content else soup.find('body')
        if not content_source:
            content_source = soup
        
        # Extract headings for structure
        headings = []
        for heading in content_source.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = heading.get_text().strip()
            if heading_text and len(heading_text) > 3:
                headings.append(f"{heading.name.upper()}: {heading_text}")
        
        # Extract paragraphs and list items
        text_elements = []
        for element in content_source.find_all(['p', 'li', 'div']):
            text = element.get_text().strip()
            if text and len(text) > 20:
                text_elements.append(text)
        
        # Extract links
        links = []
        for link in content_source.find_all('a', href=True):
            link_text = link.get_text().strip()
            link_url = link.get('href')
            if link_text and link_url and len(link_text) > 3:
                # Convert relative URLs to absolute
                if link_url.startswith('/'):
                    link_url = f"{parsed.scheme}://{parsed.netloc}{link_url}"
                links.append(f"Ã¢â‚¬Â¢ {link_text}: {link_url}")
        
        # Combine and clean all text
        all_text_parts = text_elements
        full_text = ' '.join(all_text_parts)
        
        # Advanced text cleaning
        lines = (line.strip() for line in full_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 3)
        
        # Remove excessive whitespace and normalize
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Check if we got meaningful content
        if not cleaned_text or len(cleaned_text) < 50:
            # Very little content found - might be a dynamic site or access issue
            fallback_content = (
                f"Ã¢Å¡Â Ã¯Â¸Â Limited content extracted from this page. "
                f"This could indicate:\n"
                f"Ã¢â‚¬Â¢ The page requires JavaScript to load content\n"
                f"Ã¢â‚¬Â¢ Authentication or cookies are required\n"
                f"Ã¢â‚¬Â¢ The page is a redirect or landing page\n"
                f"Ã¢â‚¬Â¢ Content is loaded via AJAX/API calls\n\n"
                f"Raw HTML size: {len(response.content)} bytes\n"
                f"Status: {response.status_code}\n"
                f"Content-Type: {content_type}"
            )
            
            return UrlAnalysisResult(
                title=title_text,
                description=description,
                keywords=keywords,
                headings=headings,
                content=fallback_content,
                links=links,
                url=url,
                success=True
            )
        
        # Limit content length
        if len(cleaned_text) > max_content_length:
            cleaned_text = cleaned_text[:max_content_length] + "..."
        
        return UrlAnalysisResult(
            title=title_text,
            description=description,
            keywords=keywords,
            headings=headings,
            content=cleaned_text,
            links=links,
            url=url,
            success=True
        )
        
    except httpx.TimeoutException:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[], 
            content="", links=[], url=url, success=False,
            error_message=f"Timeout while fetching {url} ({timeout} seconds exceeded)"
        )
    except httpx.HTTPStatusError as e:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[], 
            content="", links=[], url=url, success=False,
            error_message=f"HTTP {e.response.status_code} - {e.response.reason_phrase}"
        )
    except Exception as e:
        return UrlAnalysisResult(
            title="", description=None, keywords=None, headings=[], 
            content="", links=[], url=url, success=False,
            error_message=str(e)
        )


def fetch_url_content_sync(
    url: str, 
    max_content_length: int = 8000,
    timeout: float = 20.0
) -> UrlAnalysisResult:
    """
    Synchronous version of URL content fetching.
    
    Args:
        url: URL to fetch content from
        max_content_length: Maximum length of content to extract
        timeout: Request timeout in seconds
        
    Returns:
        UrlAnalysisResult containing extracted content and metadata
    """
    import asyncio
    import sys
    
    async def _fetch():
        async with httpx.AsyncClient() as client:
            return await fetch_url_content_async(client, url, max_content_length, timeout)
    
    # Handle the case where we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, can't use asyncio.run()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _fetch())
            return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(_fetch())
