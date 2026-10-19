"""The report of one sync run: what happened to each document, the run's problems and its exit code.

The report keeps the server URL without its userinfo and holds no token, so neither
the summary the command prints nor the JSON file it writes can carry a secret of
the command line.
"""

import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal

from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.client import strip_userinfo

type Outcome = Literal['created', 'updated', 'deleted', 'unchanged', 'kept', 'failed']
type CountKey = Literal['created', 'updated', 'deleted', 'unchanged', 'kept', 'foreign', 'failed', 'withheld']

COUNT_KEYS: tuple[CountKey, ...] = ('created', 'updated', 'deleted', 'unchanged', 'kept', 'foreign', 'failed', 'withheld')
# A configuration error outranks a transient failure, which outranks a partial run.
_SEVERITY = {EXIT_CONVERGED: 0, EXIT_PARTIAL: 1, EXIT_TRANSIENT: 2, EXIT_CONFIG: 3}
_MEANINGS = {
    EXIT_CONVERGED: 'completed',
    EXIT_PARTIAL: 'completed with problems: some documents were not synchronized',
    EXIT_CONFIG: 'stopped on a configuration, authentication or plan error',
    EXIT_TRANSIENT: 'stopped on a transient failure; rerun later',
}
# A document the run could not synchronize, or could only keep as stored, leaves the
# thread behind the corpus.
_PARTIAL_OUTCOMES: frozenset[Outcome] = frozenset({'failed', 'kept'})


@dataclass
class DocumentOutcome:
    """What the run did with one document of the corpus, or with one stored entry it removed.

    Attributes:
        source_id: The document's source id.
        outcome: What the run did.
        context_id: The stored entry the outcome concerns, when there is one.
        error: Why the document failed or was kept, when it was.
    """

    source_id: str
    outcome: Outcome
    context_id: str | None
    error: str | None


def _zero_counts() -> dict[CountKey, int]:
    """Return every count of a run that has done nothing yet."""
    return dict.fromkeys(COUNT_KEYS, 0)


@dataclass
class RunReport:
    """The outcome of one sync run.

    Attributes:
        server: The server URL without userinfo.
        thread_id: The corpus thread.
        dry_run: Whether the run only planned.
        outcomes: One outcome per document, in the order the run settled them.
        errors: Run-level problems, such as an excluded file or a directory that could not be listed.
        counts: The number of documents per outcome, the foreign members and the withheld deletes.
        exit_code: The command's exit code, the most severe one any problem called for.
    """

    server: str
    thread_id: str
    dry_run: bool
    outcomes: list[DocumentOutcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    counts: dict[CountKey, int] = field(default_factory=_zero_counts)
    exit_code: int = EXIT_CONVERGED

    def __post_init__(self) -> None:
        self.server = strip_userinfo(self.server)

    def add(self, outcome: DocumentOutcome) -> None:
        """Record a document's outcome; a failed or kept document makes the run partial."""
        self.outcomes.append(outcome)
        self.counts[outcome.outcome] += 1
        if outcome.outcome in _PARTIAL_OUTCOMES:
            self.escalate(EXIT_PARTIAL)

    def count(self, key: Literal['foreign', 'withheld'], amount: int) -> None:
        """Add to a run-level count: members another principal owns, or deletes the run did not execute."""
        self.counts[key] += amount

    def fail(self, message: str, *, exit_code: int) -> None:
        """Record a run-level problem and the exit code it calls for."""
        self.errors.append(message)
        self.escalate(exit_code)

    def escalate(self, exit_code: int) -> None:
        """Raise the run's exit code to one of the command's codes when that one is more severe.

        Args:
            exit_code: 0, 1, 2 or 69.

        Raises:
            ValueError: The code is none of the command's exit codes.
        """
        if exit_code not in _SEVERITY:
            raise ValueError(f'unknown exit code {exit_code}')
        if _SEVERITY[exit_code] > _SEVERITY[self.exit_code]:
            self.exit_code = exit_code


def render_summary(report: RunReport) -> str:
    """Return the human summary of a run: the run, every count, each document problem and the exit code.

    Args:
        report: The run's report.

    Returns:
        The summary, one line per item, without a trailing newline.
    """
    if report.dry_run:
        lines = [f'Dry run of {report.thread_id} with {report.server}: nothing was written']
    else:
        lines = [f'Sync of {report.thread_id} with {report.server}']
    lines.append(', '.join(f'{key} {report.counts[key]}' for key in COUNT_KEYS))
    lines.extend(
        f'{outcome.outcome}: {outcome.source_id}: {outcome.error}' for outcome in report.outcomes if outcome.error is not None
    )
    lines.append(f'exit {report.exit_code}: {_MEANINGS[report.exit_code]}')
    return '\n'.join(lines)


def write_json(report: RunReport, path: Path) -> None:
    """Write a run's report as a UTF-8 JSON file.

    Args:
        report: The run's report.
        path: The file to write, replaced when it exists.
    """
    payload = {
        'server': report.server,
        'thread_id': report.thread_id,
        'dry_run': report.dry_run,
        'exit_code': report.exit_code,
        'counts': dict(report.counts),
        'documents': [asdict(outcome) for outcome in report.outcomes],
        'errors': list(report.errors),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
