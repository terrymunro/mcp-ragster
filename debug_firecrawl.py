#!/usr/bin/env python3
"""Debug script to test Firecrawl API with our client."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ragster.client_firecrawl import FirecrawlAPIClient


async def test_firecrawl_client():
    """Test our Firecrawl client implementation."""

    # Test URLs to see which ones work
    test_urls = [
        "https://registry.terraform.io/providers/databricks/databricks/latest/docs",
        "https://registry.terraform.io/providers/databricks/databricks/latest",
        "https://docs.databricks.com/en/dev-tools/terraform/index.html",
    ]

    client = FirecrawlAPIClient()

    for test_url in test_urls:
        print(f"\nTesting: {test_url}")

        try:
            result = await client.crawl_url(test_url)

            print("✓ Success!")
            print(f"Content type: {result.get('type')}")
            print(f"Content length: {len(result.get('content', ''))}")

            # Show first 200 chars of content
            content = result.get("content", "")
            if content:
                print(f"Content preview: {content[:200]}...")
            break  # Stop on first success

        except Exception as e:
            print(f"✗ Error: {e}")


async def test_simple_url():
    """Test with a simpler URL first."""

    # Test URL that was failing
    test_url = "https://fastapi.tiangolo.com/tutorial/testing"
    print(f"\nTesting simple URL: {test_url}")

    try:
        client = FirecrawlAPIClient()
        result = await client.crawl_url(test_url)

        print("✓ Success!")
        print(f"Content type: {result.get('type')}")
        print(f"Content length: {len(result.get('content', ''))}")

    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_simple_url())
    asyncio.run(test_firecrawl_client())
