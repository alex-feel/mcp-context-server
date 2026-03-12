"""Summary provider implementations.

Each provider module contains a single provider class that implements
the SummaryProvider protocol using LangChain chat model integrations.

Available Providers:
- langchain_ollama.py: OllamaSummaryProvider - Local Ollama models (default)
- langchain_openai.py: OpenAISummaryProvider - OpenAI API
- langchain_anthropic.py: AnthropicSummaryProvider - Anthropic API

All providers are imported dynamically by the factory to avoid loading
unused dependencies.
"""
