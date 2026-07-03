"""Helm-template regression tests for the provider ``existingSecretKey`` override.

These render the chart with ``helm template`` and assert the rendered Deployment,
so they require the ``helm`` binary and are skipped where it is unavailable.

Background: ``embeddingSecrets`` and ``summarySecrets`` expose ``existingSecret``
but previously hardcoded the data-key NAME inside that secret (openai-api-key,
azure-openai-api-key, huggingface-api-token, voyage-api-key, anthropic-api-key),
unlike ``storage.postgresql`` and ``langsmith`` which expose a configurable
``existingSecretKey``. An operator whose secret stored the key under a different
name had no chart knob and the pod failed to start (CreateContainerConfigError).
The chart now exposes ``existingSecretKey`` for both provider-secret groups,
defaulting to the provider-specific key name when empty.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_HELM = shutil.which('helm')
pytestmark = pytest.mark.skipif(_HELM is None, reason='helm binary not available')

_CHART = Path(__file__).resolve().parents[2] / 'deploy' / 'helm' / 'mcp-context-server'


def _render(*args: str) -> str:
    assert _HELM is not None
    result = subprocess.run(
        [_HELM, 'template', 't', str(_CHART), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f'helm template failed: {result.stderr}'
    return result.stdout


def test_anthropic_existing_secret_defaults_to_provider_key_name() -> None:
    """With existingSecret set and no override, the default key name is used."""
    out = _render(
        '--set', 'search.summary.enabled=true',
        '--set', 'search.summary.provider=anthropic',
        '--set', 'summarySecrets.existingSecret=sumsec',
    )
    assert 'name: sumsec' in out
    assert 'key: anthropic-api-key' in out


def test_anthropic_existing_secret_key_override_is_honored() -> None:
    """existingSecretKey overrides the hardcoded data-key name for ANTHROPIC_API_KEY."""
    out = _render(
        '--set', 'search.summary.enabled=true',
        '--set', 'search.summary.provider=anthropic',
        '--set', 'summarySecrets.existingSecret=sumsec',
        '--set', 'summarySecrets.existingSecretKey=my-anthropic-key',
    )
    assert 'name: sumsec' in out
    assert 'key: my-anthropic-key' in out
    # The default key name must NOT be rendered once overridden.
    assert 'key: anthropic-api-key' not in out


def test_voyage_embedding_existing_secret_key_override_is_honored() -> None:
    """existingSecretKey on embeddingSecrets overrides the voyage data-key name."""
    out = _render(
        '--set', 'search.semantic.enabled=true',
        '--set', 'search.semantic.provider=voyage',
        '--set', 'embeddingSecrets.existingSecret=embsec',
        '--set', 'embeddingSecrets.existingSecretKey=my-voyage-key',
    )
    assert 'name: embsec' in out
    assert 'key: my-voyage-key' in out
    assert 'key: voyage-api-key' not in out


def test_openai_embedding_existing_secret_key_override_is_honored() -> None:
    """existingSecretKey overrides the shared openai data-key name on the embedding side."""
    out = _render(
        '--set', 'search.semantic.enabled=true',
        '--set', 'search.semantic.provider=openai',
        '--set', 'embeddingSecrets.existingSecret=embsec',
        '--set', 'embeddingSecrets.existingSecretKey=my-openai-key',
    )
    assert 'name: embsec' in out
    assert 'key: my-openai-key' in out
    assert 'key: openai-api-key' not in out
