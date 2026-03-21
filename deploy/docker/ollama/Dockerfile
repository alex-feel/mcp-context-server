ARG OLLAMA_TAG=latest
FROM ollama/ollama:${OLLAMA_TAG}

ENV EMBEDDING_MODEL=qwen3-embedding:0.6b
ENV SUMMARY_MODEL=qwen3:0.6b

COPY deploy/docker/ollama/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
