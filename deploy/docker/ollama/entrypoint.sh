#!/bin/bash
set -e

EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
SUMMARY_MODEL="${SUMMARY_MODEL:-qwen3:0.6b}"

# Start Ollama server in background on temporary internal port
OLLAMA_HOST=127.0.0.1:11155 /bin/ollama serve &
serve_pid=$!

echo "Waiting for Ollama server to start..."
until OLLAMA_HOST=127.0.0.1:11155 ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Checking if model '$EMBEDDING_MODEL' exists..."
if ! OLLAMA_HOST=127.0.0.1:11155 ollama show "$EMBEDDING_MODEL" >/dev/null 2>&1; then
  echo "Pulling model: $EMBEDDING_MODEL..."
  OLLAMA_HOST=127.0.0.1:11155 ollama pull "$EMBEDDING_MODEL"
  echo "Model pulled successfully!"
else
  echo "Model '$EMBEDDING_MODEL' already exists, skipping pull."
fi

# Pull summary model only when summary provider is ollama and model differs from embedding model
if [ "${SUMMARY_PROVIDER:-ollama}" = "ollama" ] && [ -n "$SUMMARY_MODEL" ] && [ "$SUMMARY_MODEL" != "$EMBEDDING_MODEL" ]; then
  echo "Checking if summary model '$SUMMARY_MODEL' exists..."
  if ! OLLAMA_HOST=127.0.0.1:11155 ollama show "$SUMMARY_MODEL" >/dev/null 2>&1; then
    echo "Pulling summary model: $SUMMARY_MODEL..."
    OLLAMA_HOST=127.0.0.1:11155 ollama pull "$SUMMARY_MODEL"
    echo "Summary model pulled successfully!"
  else
    echo "Summary model '$SUMMARY_MODEL' already exists, skipping pull."
  fi
else
  if [ "${SUMMARY_PROVIDER:-ollama}" != "ollama" ]; then
    echo "Summary provider is '${SUMMARY_PROVIDER}' (not ollama), skipping summary model pull."
  fi
fi

echo "Stopping temporary server..."
kill $serve_pid
wait $serve_pid 2>/dev/null || true

echo "Starting production Ollama server..."
exec /bin/ollama serve
