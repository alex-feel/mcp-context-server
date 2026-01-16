#!/bin/bash
#
# Docker entrypoint wrapper that handles exit codes to prevent infinite restart loops.
#
# How it works:
# 1. Runs the Python server
# 2. Captures the exit code
# 3. Based on exit code:
#    - 0 (success): Normal exit
#    - 78 (EX_CONFIG): Configuration error - sleep forever (NO restart)
#    - 69 (EX_UNAVAILABLE): Dependency error - exit to allow Docker restart
#    - Other: Pass through
#
# Exit code 78 (ConfigurationError) requires human intervention to fix.
# Keeping the container running (but idle) prevents infinite restart loops
# while making it easy to inspect logs and fix configuration.
#
# Exit Codes:
#   0  - Success
#   69 - Dependency error (EX_UNAVAILABLE) - may retry
#   78 - Configuration error (EX_CONFIG) - never retry
#
# Usage:
#   This script is set as the ENTRYPOINT in the Dockerfile.
#   It wraps the Python server and interprets exit codes.

# Run the server and capture exit code
# Note: We do NOT use 'set -e' because we need to capture the exit code
python -m app.server
EXIT_CODE=$?

case $EXIT_CODE in
    0)
        # Success - normal exit
        echo '[docker-entrypoint] Server exited normally (code 0)'
        exit 0
        ;;
    78)
        # EX_CONFIG: Configuration error - DO NOT restart
        echo ''
        echo '=============================================='
        echo '[FATAL] CONFIGURATION ERROR - CONTAINER HALTED'
        echo '=============================================='
        echo ''
        echo 'The server encountered a configuration error that'
        echo 'requires manual intervention. Check the logs above'
        echo 'for details.'
        echo ''
        echo 'Common causes:'
        echo '  - EMBEDDING_MODEL not found in Ollama'
        echo '  - Required environment variable not set'
        echo '  - Invalid provider configuration'
        echo '  - Missing API key'
        echo ''
        echo 'To resolve:'
        echo '  1. Fix the configuration (check logs above)'
        echo '  2. Stop this container: docker stop <container>'
        echo '  3. Start fresh: docker compose up -d'
        echo ''
        echo 'Container will remain IDLE until manually stopped.'
        echo 'This prevents infinite restart loops.'
        echo '=============================================='
        # Use exec to replace shell with sleep, making it PID 1
        # This ensures proper signal handling for container stop
        exec sleep infinity
        ;;
    69)
        # EX_UNAVAILABLE: Dependency error - MAY retry
        echo ''
        echo '[docker-entrypoint] Dependency unavailable (code 69)'
        echo '[docker-entrypoint] Container will exit - Docker may restart automatically'
        echo ''
        exit 69
        ;;
    *)
        # Other errors - pass through
        echo ''
        echo "[docker-entrypoint] Server exited with code $EXIT_CODE"
        echo ''
        exit $EXIT_CODE
        ;;
esac
