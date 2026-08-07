"""Regression tests for the CI job-skip conditions in .github/workflows.

Every quality gate (tests, pre-commit, MCP marker validation, Trivy security scan)
carries a job-level ``if:`` that exempts Release Please's own version-bump pull
request. The exemption must be authenticated by something the pull-request
submitter cannot control.

Two values were available. ``github.head_ref`` is the branch name, which a
contributor picks freely on a fork, so it is only safe when paired with an author
check. ``github.event.head_commit.message`` on the push-to-main event is worse:
depending on the repository's merge-message settings it embeds the fork branch
name, the pull-request title, or the squashed commit bodies, so it can be made to
contain any literal string a match looks for -- there is no way to authenticate the
release bot from it, and a matching push skipped pre-commit, the MCP checks and the
security scan (including its SARIF upload) on the default branch.

These tests are structural: they read the workflow files and pin the invariants, so
reintroducing an untrusted-string skip fails here rather than silently degrading the
signal a reviewer relies on. They parse the text directly rather than through a YAML
library so the suite needs no extra dependency for a check this small.
"""

import re
from pathlib import Path

import pytest

_WORKFLOW_DIR = Path(__file__).resolve().parents[2] / '.github' / 'workflows'
_WORKFLOW_FILES = sorted(_WORKFLOW_DIR.glob('*.yml'))

# An ``if:`` value in these workflows is always a GitHub expression: either the
# single-line ``if: ${{ ... }}`` form or a folded block whose last line closes with
# ``}}``. Capturing from the key to that closing brace pair yields the whole
# condition regardless of which form is used.
_CONDITION_RE = re.compile(r'^\s*if:.*?\}\}', re.DOTALL | re.MULTILINE)


def _conditions(path: Path) -> list[str]:
    """Return every job/step ``if:`` expression in one workflow file.

    Args:
        path: Workflow file to read.

    Returns:
        The raw text of each condition, in file order.
    """
    return _CONDITION_RE.findall(path.read_text(encoding='utf-8'))


def test_workflow_files_are_present() -> None:
    """Guard against the glob silently matching nothing."""
    assert _WORKFLOW_FILES, f'no workflow files found under {_WORKFLOW_DIR}'


@pytest.mark.parametrize('workflow', _WORKFLOW_FILES, ids=lambda p: p.name)
def test_no_job_gates_on_the_commit_message(workflow: Path) -> None:
    """No job may decide whether to run by matching the pushed commit message."""
    text = workflow.read_text(encoding='utf-8')
    assert 'head_commit.message' not in text, (
        f'{workflow.name} gates on the commit message, which a contributor can control '
        f'through the fork branch name, the pull-request title or a squashed commit body'
    )


@pytest.mark.parametrize('workflow', _WORKFLOW_FILES, ids=lambda p: p.name)
def test_branch_prefix_skips_are_paired_with_an_author_check(workflow: Path) -> None:
    """A skip keyed on the branch name must also verify who opened the pull request."""
    for condition in _conditions(workflow):
        if 'head_ref' not in condition:
            continue
        assert 'github.event.pull_request.user.login' in condition, (
            f'{workflow.name} skips on the head branch name alone; a contributor names a '
            f'fork branch freely, so the exemption must also require the pull-request '
            f'author to be the trusted release identity'
        )
        assert 'github.repository_owner' in condition, (
            f'{workflow.name} compares the pull-request author against something other '
            f'than the repository owner, which is the identity Release Please runs as'
        )


def test_the_quality_gates_still_carry_a_skip_condition() -> None:
    """The two gate workflows must keep an authenticated exemption, not lose it entirely.

    Without this the previous test passes vacuously if every condition is deleted.
    """
    for name in ('lint.yml', 'test.yml'):
        path = _WORKFLOW_DIR / name
        assert path.exists(), f'{name} is missing'
        conditions = [c for c in _conditions(path) if 'head_ref' in c]
        assert conditions, f'{name} no longer exempts the release bump pull request'
