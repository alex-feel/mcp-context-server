"""Operator console output and settings loading for the compression CLI."""

import sys

from pydantic import ValidationError

from app.compression.types import CompressionMetadata
from app.settings import AppSettings
from app.settings import get_settings

# Default probe-batch size used to estimate total runtime in the dry-run
# preview. Small enough to keep dry-run cheap while large enough to
# smooth out single-call jitter.
PROBE_BATCH_SIZE = 128

# Warning text printed before any destructive operation. Plain ASCII so
# Windows cmd, PowerShell, and POSIX shells all render the same.
WARNING_BORDER = '=' * 60
WARNING_RULE = '-' * 60


def print_warning(
    *,
    source_url: str,
    mode: str,
    mode_description: str,
    dry_run: bool,
) -> None:
    """Print the BACKUP REQUIRED warning block to stderr.

    Args:
        source_url: URL of the source database (already credential-masked).
        mode: Short mode flag (``--compress`` or ``--decompress``).
        mode_description: Human-readable description of the operation.
        dry_run: Append a DRY-RUN line when True.
    """
    lines = [
        WARNING_BORDER,
        'BACKUP REQUIRED -- THIS OPERATION IS NOT REVERSIBLE WITHOUT',
        'YOUR OWN BACKUP. The source vector table will be permanently',
        'modified by this run.',
        WARNING_RULE,
        f'Source: {source_url}',
        f'Mode:   {mode} ({mode_description})',
    ]
    if dry_run:
        lines.append('DRY-RUN: no writes will be issued')
    lines.append(WARNING_BORDER)
    print('\n'.join(lines), file=sys.stderr)


def print_plan(
    *,
    source_url: str,
    from_table: str,
    from_rows: int,
    to_table: str,
    provenance: CompressionMetadata,
    estimated_seconds: float,
    estimated_rate: float,
) -> None:
    """Print the structured migration plan to stderr."""
    lines = [
        '',
        'Plan',
        f'  source:        {source_url}',
        f'  from_table:    {from_table} ({from_rows} rows)',
        f'  to_table:      {to_table}',
        (
            f'  provenance:    compression_metadata (singleton; '
            f'bits={provenance.bits} variant={provenance.variant} '
            f'seed={provenance.seed} dim={provenance.dim})'
        ),
        (
            f'  estimated_time: {estimated_seconds:.2f} seconds at '
            f'{estimated_rate:.0f} rows/s (extrapolated from a '
            f'{PROBE_BATCH_SIZE}-row probe)'
        ),
    ]
    print('\n'.join(lines), file=sys.stderr)


def load_cli_settings() -> AppSettings | None:
    """Resolve settings for a CLI run, mapping env misconfiguration to a clean error.

    ``get_settings()`` validates the WHOLE environment and raises a pydantic
    ``ValidationError`` on any invalid combination. The documented
    ``--decompress`` prerequisite (unset ``ENABLE_EMBEDDING_COMPRESSION``)
    recreates exactly the env shape the fp32 pgvector dimension validator
    rejects on a PostgreSQL deployment whose EMBEDDING_DIM exceeds the index
    cap, so a direct invocation of :func:`run_compress`/:func:`run_decompress`
    would die with a raw multi-line validation traceback instead of an
    operator-facing CLI message. Condense the validation detail (field name
    plus message per error) to one line on stderr; callers return
    ``ConfigurationError.EXIT_CODE`` (78, EX_CONFIG) -- the same permanent-
    misconfiguration classification ``main()`` applies when the same
    ``ValidationError`` surfaces at module-import time.

    Returns:
        The resolved settings, or ``None`` when the environment is invalid
        (the error detail has already been printed to stderr).
    """
    try:
        return get_settings()
    except ValidationError as exc:
        parts: list[str] = []
        for err in exc.errors():
            loc = '.'.join(str(piece) for piece in err['loc'])
            msg = str(err['msg'])
            parts.append(f'{loc}: {msg}' if loc else msg)
        print(
            f'[ERROR] Configuration invalid: {"; ".join(parts)}',
            file=sys.stderr,
        )
        return None
