import os

# Set dummy environment variables so ragster.config can be imported
os.environ.setdefault("VOYAGEAI_API_KEY", "test")
os.environ.setdefault("PERPLEXITY_API_KEY", "test")
os.environ.setdefault("JINA_API_KEY", "test")
os.environ.setdefault("FIRECRAWL_API_URL", "http://localhost")

