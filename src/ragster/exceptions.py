class MCPError(Exception):
    """Base class for exceptions in this application."""

    def __init__(self, message: str = "An MCP error occurred", status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def __str__(self):
        return f"[Status Code: {self.status_code}] {self.message}"


class EmbeddingServiceError(MCPError):
    """Custom exception for embedding service failures."""

    def __init__(
        self,
        message: str = "Embedding service operation failed",
        status_code: int = 503,
        underlying_error: Exception | None = None,
    ):
        super().__init__(message, status_code)
        self.underlying_error = underlying_error
        if underlying_error:
            self.message = f"{message}: {str(underlying_error)}"


class MilvusOperationError(MCPError):
    """Custom exception for Milvus operation failures."""

    def __init__(
        self,
        message: str = "Milvus operation failed",
        status_code: int = 503,
        underlying_error: Exception | None = None,
    ):
        super().__init__(message, status_code)
        self.underlying_error = underlying_error
        if underlying_error:
            self.message = f"{message}: {str(underlying_error)}"


class ExternalAPIError(MCPError):
    """Base class for external API errors."""

    def __init__(
        self,
        service_name: str,
        message: str = "External API call failed",
        status_code: int = 502,
        underlying_error: Exception | None = None,
    ):
        full_message = f"{service_name} API Error: {message}"
        if underlying_error:
            full_message = f"{full_message}: {str(underlying_error)}"
        super().__init__(full_message, status_code)
        self.service_name = service_name
        self.underlying_error = underlying_error


class JinaAPIError(ExternalAPIError):
    """Custom exception for Jina API failures."""

    def __init__(
        self,
        message: str = "Jina API call failed",
        status_code: int = 502,
        underlying_error: Exception | None = None,
    ):
        super().__init__("Jina", message, status_code, underlying_error)


class PerplexityAPIError(ExternalAPIError):
    """Custom exception for Perplexity API failures."""

    def __init__(
        self,
        message: str = "Perplexity API call failed",
        status_code: int = 502,
        underlying_error: Exception | None = None,
    ):
        super().__init__("Perplexity", message, status_code, underlying_error)


class FirecrawlError(ExternalAPIError):
    """Custom exception for Firecrawl operation failures."""

    def __init__(
        self,
        message: str = "Firecrawl operation failed",
        status_code: int = 502,
        underlying_error: Exception | None = None,
    ):
        super().__init__("Firecrawl", message, status_code, underlying_error)


class ConfigurationError(ValueError, MCPError):
    """Custom exception for configuration problems.
    Typically raised at startup, so status_code might be less relevant for HTTP response.
    """

    def __init__(self, message: str = "Configuration error"):
        # For config errors, status_code might not be directly used in HTTP response if app fails to start
        super().__init__(message, status_code=500)  # MCPError part
        ValueError.__init__(self, message)  # ValueError part


class APICallError(MCPError):
    """Generic error for when an API call fails and we want to pass status through."""

    def __init__(self, service_name: str, http_status: int, detail: str | None = None):
        message = f"Error calling {service_name} API: HTTP {http_status}"
        if detail:
            message += f" - {detail}"
        # Try to map common HTTP errors to client/server side for status_code
        # For external services, a 5xx from them is a 502 Bad Gateway for us.
        # A 4xx from them might be a 400 Bad Request from us if we passed bad data,
        # or a 502 if it's an issue like auth failure on their end (which is our config issue).
        # For simplicity, let's use 502 for most upstream issues we can't control.
        super().__init__(
            message,
            status_code=502
            if http_status >= 500
            else (400 if http_status in [400, 401, 403] else 500),
        )
        self.service_name = service_name
        self.http_status = http_status
        self.detail = detail
