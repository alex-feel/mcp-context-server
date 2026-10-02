"""Settings shared across provider layers: the Ollama connection and LangSmith tracing."""

from pydantic import Field
from pydantic import SecretStr

from app.settings.base import CommonSettings


class OllamaSettings(CommonSettings):
    """Shared Ollama infrastructure settings.

    Contains settings shared by all features that use Ollama
    (embeddings, summary generation).
    """

    host: str = Field(
        default='http://localhost:11434',
        alias='OLLAMA_HOST',
        description='Ollama server URL',
    )
    auto_pull: bool = Field(
        default=True,
        alias='OLLAMA_AUTO_PULL',
        description='Automatically pull missing Ollama models on startup',
    )
    pull_timeout: int = Field(
        default=900,
        alias='OLLAMA_PULL_TIMEOUT_S',
        ge=30,
        le=3600,
        description='Timeout in seconds for pulling Ollama models (default: 900s for slow networks)',
    )


class LangSmithSettings(CommonSettings):
    """LangSmith tracing settings for cost tracking and observability."""

    tracing: bool = Field(
        default=False,
        alias='LANGSMITH_TRACING',
        description='Enable LangSmith tracing for cost tracking and observability',
    )
    api_key: SecretStr | None = Field(
        default=None,
        alias='LANGSMITH_API_KEY',
        description='LangSmith API key for tracing',
    )
    endpoint: str = Field(
        default='https://api.smith.langchain.com',
        alias='LANGSMITH_ENDPOINT',
        description='LangSmith API endpoint',
    )
    project: str = Field(
        default='mcp-context-server',
        alias='LANGSMITH_PROJECT',
        description='LangSmith project name for grouping traces',
    )
