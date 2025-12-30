#!/bin/bash
set -e

MODEL="${MODEL:-embeddinggemma:latest}"

# Start Ollama server in background on temporary internal port
OLLAMA_HOST=127.0.0.1:11155 /bin/ollama serve &
serve_pid=$!

echo "Waiting for Ollama server to start..."
until OLLAMA_HOST=127.0.0.1:11155 ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Checking if model '$MODEL' exists..."
if ! OLLAMA_HOST=127.0.0.1:11155 ollama list | grep -q "${MODEL%%:*}"; then
  echo "Pulling model: $MODEL..."
  OLLAMA_HOST=127.0.0.1:11155 ollama pull "$MODEL"
  echo "Model pulled successfully!"
else
  echo "Model '$MODEL' already exists, skipping pull."
fi

echo "Stopping temporary server..."
kill $serve_pid
wait $serve_pid 2>/dev/null || true

echo "Starting production Ollama server..."
exec /bin/ollama serve
