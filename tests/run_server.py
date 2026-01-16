#!/usr/bin/env python
"""Wrapper to run the MCP server with proper Python path and environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def is_ollama_model_available(model: str, host: str = 'http://localhost:11434') -> bool:
    """Check if a specific Ollama model is available.

    Performs two checks:
    1. Ollama service is running at the specified host
    2. The specified model is installed and available

    Args:
        model: Model name (e.g., 'qwen3-embedding:0.6b', 'all-minilm')
        host: Ollama host URL (default: http://localhost:11434)

    Returns:
        True if model is available, False otherwise
    """
    try:
        import httpx

        # Check 1: Service is running (short timeout to not slow down tests)
        with httpx.Client(timeout=2.0) as client:
            response = client.get(host)
            if response.status_code != 200:
                return False

        # Check 2: Model is available
        import ollama

        ollama_client = ollama.Client(host=host, timeout=5.0)
        ollama_client.show(model)
        return True

    except ImportError:
        # ollama or httpx package not installed
        return False
    except Exception:
        # Service not running or model not available
        return False


# Force test mode for all test runs
# Check if we're being run from pytest or in a test context
if 'pytest' in sys.modules or any('test' in arg.lower() for arg in sys.argv):
    # We're in a test context - use temporary database
    # Note: FastMCP Client spawns subprocesses without inheriting environment,
    # so we create our own temp database and enable semantic search
    import tempfile

    # Only create a temp DB if one wasn't provided by the parent process
    if 'DB_PATH' not in os.environ:
        temp_dir = tempfile.mkdtemp(prefix='mcp_server_wrapper_')
        test_db = Path(temp_dir) / 'test_wrapper.db'
        os.environ['DB_PATH'] = str(test_db)
    else:
        test_db = Path(os.environ['DB_PATH'])

    os.environ['MCP_TEST_MODE'] = '1'

    # Only set these if not already set by parent process
    # This allows tests to control these settings
    if 'ENABLE_SEMANTIC_SEARCH' not in os.environ:
        os.environ['ENABLE_SEMANTIC_SEARCH'] = 'true'
    if 'ENABLE_FTS' not in os.environ:
        os.environ['ENABLE_FTS'] = 'true'
    if 'ENABLE_HYBRID_SEARCH' not in os.environ:
        os.environ['ENABLE_HYBRID_SEARCH'] = 'true'

    # DISABLED_TOOLS is passed through from parent process if set
    # This allows tests to control which tools are disabled

    # Smart Embedding Configuration
    # This implements the user's requirement: "Tests MUST enable embedding generation
    # when model IS available, and only disable when model is NOT available"
    #
    # Priority order for model detection:
    # 1. all-minilm (CI model - small, fast)
    # 2. qwen3-embedding:0.6b (default production model)
    #
    # If EMBEDDING_MODEL is already set by parent (e.g., CI), check if it's available
    # If not set, detect what's available and configure accordingly
    embedding_model = os.environ.get('EMBEDDING_MODEL')
    embedding_dim = os.environ.get('EMBEDDING_DIM')

    if embedding_model is None:
        # No model specified by parent - detect what's available
        candidate_models = [
            ('all-minilm', '384'),  # CI model (lightweight)
            ('qwen3-embedding:0.6b', '1024'),  # Default production model
        ]

        model_available = False
        for model, dim in candidate_models:
            if is_ollama_model_available(model):
                os.environ['EMBEDDING_MODEL'] = model
                os.environ['EMBEDDING_DIM'] = dim
                model_available = True
                print(f'[TEST SERVER] Detected available model: {model} (dim={dim})', file=sys.stderr)
                break

        if not model_available:
            # No model available - disable embedding generation
            os.environ['ENABLE_EMBEDDING_GENERATION'] = 'false'
            os.environ['ENABLE_SEMANTIC_SEARCH'] = 'false'
            print('[TEST SERVER] No Ollama model available - disabling embedding generation', file=sys.stderr)
    else:
        # Model explicitly specified (e.g., by CI) - verify it's available
        if not is_ollama_model_available(embedding_model):
            os.environ['ENABLE_EMBEDDING_GENERATION'] = 'false'
            os.environ['ENABLE_SEMANTIC_SEARCH'] = 'false'
            print(
                f'[TEST SERVER] Specified model "{embedding_model}" not available - disabling embedding generation',
                file=sys.stderr,
            )
        else:
            # Model is available, ensure DIM is set
            if embedding_dim is None:
                # Default dimensions for known models
                known_dims = {
                    'all-minilm': '384',
                    'qwen3-embedding:0.6b': '1024',
                }
                os.environ['EMBEDDING_DIM'] = known_dims.get(embedding_model, '1024')
            print(
                f'[TEST SERVER] Using specified model: {embedding_model} (dim={os.environ.get("EMBEDDING_DIM")})',
                file=sys.stderr,
            )

    print(f'[TEST SERVER] Test mode with DB_PATH={test_db}', file=sys.stderr)
    enable_emb_gen = os.environ.get('ENABLE_EMBEDDING_GENERATION', 'true')
    print(f'[TEST SERVER] ENABLE_EMBEDDING_GENERATION={enable_emb_gen}', file=sys.stderr)
    print(f'[TEST SERVER] ENABLE_SEMANTIC_SEARCH={os.environ.get("ENABLE_SEMANTIC_SEARCH")}', file=sys.stderr)
    print(f'[TEST SERVER] ENABLE_FTS={os.environ.get("ENABLE_FTS")}', file=sys.stderr)
    print(f'[TEST SERVER] ENABLE_HYBRID_SEARCH={os.environ.get("ENABLE_HYBRID_SEARCH")}', file=sys.stderr)
    if 'EMBEDDING_MODEL' in os.environ:
        print(f'[TEST SERVER] EMBEDDING_MODEL={os.environ["EMBEDDING_MODEL"]}', file=sys.stderr)
        print(f'[TEST SERVER] EMBEDDING_DIM={os.environ.get("EMBEDDING_DIM", "not set")}', file=sys.stderr)
    if 'DISABLED_TOOLS' in os.environ:
        print(f'[TEST SERVER] DISABLED_TOOLS={os.environ["DISABLED_TOOLS"]}', file=sys.stderr)

    # Double-check we're not using the default database
    default_db = Path.home() / '.mcp' / 'context_storage.db'
    if test_db.resolve() == default_db.resolve():
        raise RuntimeError(
            f'CRITICAL: Test server attempting to use default database!\nDefault: {default_db}\nDB_PATH: {test_db}',
        )
else:
    # Normal mode - check environment
    if os.environ.get('MCP_TEST_MODE') == '1':
        db_path = os.environ.get('DB_PATH')
        if db_path:
            print(f'[TEST SERVER] Running in test mode with DB_PATH={db_path}', file=sys.stderr)

            # Double-check we're not using the default database
            default_db = Path.home() / '.mcp' / 'context_storage.db'
            if Path(db_path).resolve() == default_db.resolve():
                raise RuntimeError(
                    f'CRITICAL: Test server attempting to use default database!\nDefault: {default_db}\nDB_PATH: {db_path}',
                )
        else:
            print('[TEST SERVER] WARNING: MCP_TEST_MODE=1 but DB_PATH not set!', file=sys.stderr)
    else:
        print('[TEST SERVER] Running in normal mode', file=sys.stderr)

# Now import and run the server
if __name__ == '__main__':
    from app.server import main

    # Run the server's main function
    # The server will use DB_PATH from environment via settings.py
    main()
