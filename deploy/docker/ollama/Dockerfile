FROM ollama/ollama:latest

ENV EMBEDDING_MODEL=qwen3-embedding:0.6b
ENV SUMMARY_MODEL=qwen3:1.7b

COPY deploy/docker/ollama/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
