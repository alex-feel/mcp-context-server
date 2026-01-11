"""
Embedding provider implementations.

Each provider module contains a single provider class that implements
the EmbeddingProvider protocol using LangChain integrations.

Available Providers:
- langchain_ollama.py: OllamaEmbeddingProvider - Local Ollama models
- langchain_openai.py: OpenAIEmbeddingProvider - OpenAI API
- langchain_azure.py: AzureEmbeddingProvider - Azure OpenAI Service
- langchain_huggingface.py: HuggingFaceEmbeddingProvider - HuggingFace Hub
- langchain_voyage.py: VoyageEmbeddingProvider - Voyage AI API

All providers are imported dynamically by the factory to avoid loading
unused dependencies.
"""
