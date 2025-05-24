# Model Context Protocol (MCP) RAG Server (using mcp-sdk)

This server provides a FastAPI-based implementation for a Model Context Protocol, built using the `modelcontextprotocol/python-sdk`.
It is designed to enhance Retrieval Augmented Generation (RAG) pipelines by using external tools like Jina AI (for web search), Firecrawl (for web crawling), Perplexity AI (for summaries), Milvus (as a vector store), and Voyage AI (for embeddings) to gather and retrieve information about specified topics. API keys for these services are mandatory for the server to run.

## Features

Built using the `mcp` Python SDK (`FastMCP`):

- **`load_topic_context` tool**:
  - Fetches relevant URLs and snippets for a topic using Jina AI.
  - Indexes these snippets directly into Milvus using Voyage AI embeddings.
  - Concurrently queries Perplexity AI for a summary and indexes it.
  - Concurrently crawls Jina URLs using Firecrawl and indexes content.
- **`query_topic_context` tool**:
  - Accepts a query string and `top_k`.
  - Embeds the query using Voyage AI.
  - Searches the Milvus vector store for relevant text fragments.
- **Lifespan Management**: Initializes and cleans up external service clients (Voyage AI, Milvus, etc.).
- **Robust Error Handling**: Uses custom exceptions.

## Project Structure

```sh
mcp_rag_server/
├── pyproject.toml        # Project definition and dependencies for uv
├── .env.example          # Example environment variables
├── README.md             # This file
└── src/
├── init.py
├── mcp_app.py        # Main MCP server application using FastMCP
├── config.py         # Configuration loading
├── models.py         # Pydantic models for API data structures
├── exceptions.py     # Custom exception classes
├── embedding_client.py # Handles text embedding via Voyage AI
├── milvus_ops.py     # Milvus operations
└── external_apis.py  # Clients for Jina, Firecrawl, Perplexity
```

## Prerequisites

- Python 3.10 - 3.13
- `uv` package manager installed (`pip install uv`)
- Access to a running Milvus instance.
- Access to a self-hosted Firecrawl instance or a Firecrawl API key.
- API keys for Voyage AI, Perplexity AI, and Jina AI (these are mandatory).

## Setup

1. **Create Project Files:**
   Create the files and directories as listed. Copy the content for each file.

2. **Create a virtual environment and install dependencies:**

   ```bash
   cd path/to/mcp_rag_server
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install "mcp[cli]>=0.2.0" # Installs the MCP SDK with CLI tools
   uv pip install .               # Installs other dependencies from pyproject.toml
   ```

3. **Configure Environment Variables:**
   Copy `.env.example` to `.env`. Edit `.env` with your actual API keys, Milvus details, etc. All API keys are mandatory. Ensure `EMBEDDING_DIMENSION` matches your `VOYAGEAI_MODEL_NAME`.

## Running the Server

You can run the server in development mode using the `mcp` CLI:

```bash
mcp dev src/mcp_app.py
```

This will typically start the server and provide an inspector interface.

Alternatively, for direct execution (if you want to manage Uvicorn yourself or for production without the mcp dev wrapper, though mcp provides production deployment guidance):

```bash
python src/mcp_app.py
```

This relies on the mcp_server.run() call within src/mcp_app.py.

The server will expose its tools according to the Model Context Protocol. The exact endpoint (e.g., /mcp, /sse) depends on the transport configured or defaulted by the mcp SDK.

## MCP Tools Exposed

1. `load_topic_context`

   Loads and indexes information about a topic.

   - **Input Arguments (as JSON for the tool call):**

     ```JSON
     {
         "topic": "Advancements in Quantum Computing"
     }
     ```

   - **Output (SDKLoadTopicResponse model):**

     ```JSON
     {
         "message": "Topic 'Advancements in Quantum Computing' processing initiated. All operations initiated successfully.",
         "topic": "Advancements in Quantum Computing",
         "urls_processed": 3,
         "perplexity_queried": true,
         "jina_results_found": 5,
         "errors": []
     }
     ```

2. `query_topic_context`

   Queries the indexed information.

   - **Input Arguments (as JSON for the tool call):**

   ```JSON
   {
       "query": "What are the main challenges in building stable qubits?",
       "top_k": 3
   }
   ```

   - **Output (SDKQueryTopicResponse model):**

   ```json
   {
     "query": "What are the main challenges in building stable qubits?",
     "results": [
       {
         "id": "450123...",
         "text_content": "Decoherence remains a major hurdle...",
         "source_type": "firecrawl",
         "source_identifier": "[http://example.com/quantum_challenges](http://example.com/quantum_challenges)",
         "topic": "Advancements in Quantum Computing",
         "distance": 0.85
       }
     ],
     "message": "Found 1 relevant documents."
   }
   ```

Refer to the MCP Python SDK documentation for details on how clients interact with MCP tools.

## Development Notes

- Built with the modelcontextprotocol/python-sdk.
- Embeddings: Voyage AI exclusively.
- API Keys: All external service API keys are mandatory and checked at startup via config.py.
- Milvus: HNSW index parameters are configurable via .env.
- Error Handling: Uses custom exceptions; tools return structured responses that include error information or exceptions are handled by FastMCP's underlying FastAPI.
