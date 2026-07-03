"""Helm-template regression tests for OpenAI API-key secret injection.

These render the chart with ``helm template`` and assert the rendered manifests, so they require
the ``helm`` binary and are skipped where it is unavailable (mirroring the docker-gated
integration tests). They lock the fix that unified the openai embedding and openai summary keys
to a single ``OPENAI_API_KEY`` env var and a single ``openai-api-key`` Secret data key: a
dual-openai config previously rendered TWO ``OPENAI_API_KEY`` env entries, which Kubernetes
resolves last-wins to a frequently-absent key, breaking pod startup.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_HELM = shutil.which('helm')
pytestmark = pytest.mark.skipif(_HELM is None, reason='helm binary not available')

_CHART = Path(__file__).resolve().parents[2] / 'deploy' / 'helm' / 'mcp-context-server'

_DUAL_OPENAI = (
    '--set', 'search.semantic.enabled=true',
    '--set', 'search.semantic.provider=openai',
    '--set', 'search.summary.enabled=true',
    '--set', 'search.summary.provider=openai',
)


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


def _env_count(out: str, name: str) -> int:
    """Count container env entries named exactly ``name`` (avoids AZURE_OPENAI_API_KEY overlap)."""
    return out.count(f'- name: {name}')


def _data_key_count(out: str, key: str) -> int:
    """Count Secret data lines whose key is exactly ``key`` (stripped prefix match)."""
    return sum(1 for line in out.splitlines() if line.strip().startswith(f'{key}:'))


@pytest.mark.parametrize(
    'profile',
    [
        (),
        ('-f', str(_CHART / 'values-sqlite.yaml')),
        ('-f', str(_CHART / 'values-postgresql.yaml'), '--set', 'storage.postgresql.password=p'),
    ],
    ids=['default', 'sqlite', 'postgresql'],
)
def test_dual_openai_renders_single_key(profile: tuple[str, ...]) -> None:
    """openai embedding + openai summary -> exactly one OPENAI_API_KEY env and one secret key."""
    out = _render(*profile, *_DUAL_OPENAI, '--set', 'embeddingSecrets.openaiApiKey=K')
    assert _env_count(out, 'OPENAI_API_KEY') == 1
    assert _data_key_count(out, 'openai-api-key') == 1
    assert _data_key_count(out, 'summary-openai-api-key') == 0


def test_dual_openai_summary_key_only_still_populates() -> None:
    """Dual-openai with only summarySecrets set still yields one env + one populated key.

    This is the case the old template happened to render (via last-wins to the present summary
    key); the unified fix must keep it working rather than regress it to a dangling reference.
    """
    out = _render(*_DUAL_OPENAI, '--set', 'summarySecrets.openaiApiKey=K')
    assert _env_count(out, 'OPENAI_API_KEY') == 1
    assert _data_key_count(out, 'openai-api-key') == 1


def test_dual_openai_existing_secret_precedence() -> None:
    """Dual-openai with an embedding existingSecret references it and generates no secret key."""
    out = _render(*_DUAL_OPENAI, '--set', 'embeddingSecrets.existingSecret=embsec')
    assert _env_count(out, 'OPENAI_API_KEY') == 1
    assert 'name: embsec' in out
    assert _data_key_count(out, 'openai-api-key') == 0


def test_dual_openai_embedding_existingsecret_summary_inline_no_orphan() -> None:
    """Mixed split (emb existingSecret + summary inline) refs the existingSecret, no orphan key.

    Both openai providers read the single OPENAI_API_KEY, so when the embedding
    side supplies an existingSecret the Deployment points OPENAI_API_KEY at it.
    The chart-managed Secret must NOT also render an orphaned, never-referenced
    ``openai-api-key`` from the summary inline value -- the Deployment's name
    selection and the Secret's key population must stay aligned.
    """
    out = _render(
        *_DUAL_OPENAI,
        '--set', 'embeddingSecrets.existingSecret=embsec',
        '--set', 'summarySecrets.openaiApiKey=SUMKEY',
    )
    assert _env_count(out, 'OPENAI_API_KEY') == 1
    assert 'name: embsec' in out
    assert _data_key_count(out, 'openai-api-key') == 0


def test_dual_openai_summary_existingsecret_embedding_inline_no_orphan() -> None:
    """Mixed split (summary existingSecret + embedding inline) refs the existingSecret, no orphan key.

    Symmetric counterpart: the summary side supplies an existingSecret while the
    embedding side supplies an inline key. The Deployment references the summary
    existingSecret and the chart Secret renders no orphaned ``openai-api-key``.
    """
    out = _render(
        *_DUAL_OPENAI,
        '--set', 'summarySecrets.existingSecret=sumsec',
        '--set', 'embeddingSecrets.openaiApiKey=EMBKEY',
    )
    assert _env_count(out, 'OPENAI_API_KEY') == 1
    assert 'name: sumsec' in out
    assert _data_key_count(out, 'openai-api-key') == 0


def test_openai_embedding_anthropic_summary_not_over_suppressed() -> None:
    """Mixed openai-embedding + anthropic-summary keeps both distinct keys (one each)."""
    out = _render(
        '--set', 'search.semantic.enabled=true',
        '--set', 'search.semantic.provider=openai',
        '--set', 'search.summary.enabled=true',
        '--set', 'search.summary.provider=anthropic',
        '--set', 'embeddingSecrets.openaiApiKey=E',
        '--set', 'summarySecrets.anthropicApiKey=A',
    )
    assert _env_count(out, 'OPENAI_API_KEY') == 1
    assert _env_count(out, 'ANTHROPIC_API_KEY') == 1
