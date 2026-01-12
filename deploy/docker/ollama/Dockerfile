FROM ollama/ollama:latest

ENV MODEL=qwen3-embedding:0.6b

COPY deploy/docker/ollama/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
