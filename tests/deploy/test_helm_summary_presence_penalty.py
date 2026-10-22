"""Helm-template tests for the optional Ollama summary presence penalty.

The chart renders ``SUMMARY_OLLAMA_PRESENCE_PENALTY`` only when a value is set, because the
server derives the penalty from the model family when the variable is absent and an empty
string would fail its validation. These tests render the chart with ``helm template`` and are
skipped where the ``helm`` binary is unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_HELM = shutil.which('helm')
pytestmark = pytest.mark.skipif(_HELM is None, reason='helm binary not available')

_CHART = Path(__file__).resolve().parents[2] / 'deploy' / 'helm' / 'mcp-context-server'

_OLLAMA_SUMMARY = (
    '--set', 'ollama.enabled=true',
    '--set', 'search.summary.enabled=true',
    '--set', 'search.summary.provider=ollama',
)


def _render(*args: str) -> str:
    assert _HELM is not None
    result = subprocess.run(
        [_HELM, 'template', 't', str(_CHART), *_OLLAMA_SUMMARY, *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f'helm template failed: {result.stderr}'
    return result.stdout


def _config_value(out: str, name: str) -> str | None:
    """Return the rendered ConfigMap value of ``name``, or None when the key is absent."""
    for line in out.splitlines():
        key, sep, value = line.strip().partition(': ')
        if sep and key == name:
            return value
    return None


def test_unset_penalty_renders_no_variable() -> None:
    """Left empty, the chart renders no variable, so the server applies its model-family default."""
    out = _render()

    assert _config_value(out, 'SUMMARY_OLLAMA_PRESENCE_PENALTY') is None
    assert _config_value(out, 'SUMMARY_OLLAMA_NUM_CTX') == '"32768"'


@pytest.mark.parametrize(('value', 'rendered'), [('0.7', '"0.7"'), ('0', '"0"'), ('2', '"2"')])
def test_set_penalty_renders_the_value(value: str, rendered: str) -> None:
    """A set value, zero included, reaches the ConfigMap as a quoted string."""
    out = _render('--set', f'ollama.summaryPresencePenalty={value}')

    assert _config_value(out, 'SUMMARY_OLLAMA_PRESENCE_PENALTY') == rendered
