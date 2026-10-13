"""Remove the PostgreSQL row-level security this server installed.

Implements the ``--disable-rls`` flag of ``mcp-context-server-migrate``. It connects with
the ``--source-url`` credentials, which must be the table owner's, and drops, under the
schema-init lock and in one transaction, the read views and their write rejection, every
policy of ours, the guard trigger and the helper functions. It disables row-level
security only on the tables the server enabled, and reports every table it leaves
enabled. ``--dry-run`` prints the same report and changes nothing.

Every server instance must be stopped first: a server started afterwards with
``ENABLE_POSTGRESQL_RLS=true`` installs the objects again, and one started with the flag
off before the removal is refused. The removal refuses while a session of a member of the
policy role is connected to the database; ``--force`` removes them anyway.

The CLI is single-backend: the source is the database under operation and
``--target-url`` is ignored (mirrors ``--compress``, ``--decompress``, ``--re-embed``,
``--embed-missing`` and ``--reassign-owner``).
"""

import asyncio
import logging
import sys

from app.cli._backend import assert_maintenance_connection
from app.cli._backend import make_backend
from app.cli._backend import owner_refusal
from app.cli._backend import shutdown_backend
from app.cli._database_url import mask_credentials
from app.migrations.rls_removal import MemberSessionsConnectedError
from app.migrations.rls_removal import RlsRemovalReport
from app.migrations.rls_removal import describe_sessions
from app.migrations.rls_removal import remove_rls
from app.settings import get_settings

logger = logging.getLogger(__name__)


def run_disable_rls(source_url: str, *, dry_run: bool, force: bool) -> int:
    """Public entry point for the ``--disable-rls`` flag.

    Args:
        source_url: Database URL passed to ``--source-url``, with the table owner's credentials.
        dry_run: When True, report what would be removed; change nothing.
        force: Remove even while members of the policy role are connected.

    Returns:
        Process exit code: 0 on success, also when nothing is installed; 1 while members
        of the policy role are connected and ``force`` is off; 2 on an unrecoverable
        failure. A connection row-level security filters, or a privilege error, is not a
        failure of this mode: it propagates to the dispatcher, which prints the owner
        remediation and exits 1.
    """
    # Resolve the settings before the async body so an invalid environment reaches the
    # dispatcher, which exits EX_CONFIG (78), instead of the generic failure path below.
    get_settings()

    lines = [
        'Removing the row-level security this server installed',
        f'Source: {mask_credentials(source_url)}',
        'Stop every server instance before running.',
    ]
    if dry_run:
        lines.append('DRY-RUN: nothing is dropped or disabled; the report shows what a real run does')
    print('\n'.join(lines), file=sys.stderr)

    try:
        report = asyncio.run(_disable_rls_async(source_url, dry_run=dry_run, force=force))
    except MemberSessionsConnectedError as exc:
        print(
            f'[ERROR] Refusing to remove row-level security: {describe_sessions(exc.sessions)}. '
            'Stop every server instance first, or rerun with --force to remove it while they stay connected.',
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        if owner_refusal(exc) is not None:
            raise
        logger.exception('row-level security removal failed: %s', exc)
        return 2
    _print_report(report, dry_run=dry_run)
    return 0


async def _disable_rls_async(source_url: str, *, dry_run: bool, force: bool) -> RlsRemovalReport:
    """Async body for :func:`run_disable_rls`.

    Args:
        source_url: URL passed to ``--source-url``.
        dry_run: When True, report only.
        force: Remove even while members of the policy role are connected.

    Returns:
        What was removed, or what a dry run would remove.
    """
    settings = get_settings()
    # Removal touches no vector column, so the pgvector extension and its codec are never
    # provisioned for this mode.
    backend = make_backend(source_url, provision_vector=False)
    await backend.initialize()
    try:
        await assert_maintenance_connection(backend, settings.storage.postgresql_schema)
        return await remove_rls(backend, settings, dry_run=dry_run, force=force)
    finally:
        await shutdown_backend(backend)


def _print_report(report: RlsRemovalReport, *, dry_run: bool) -> None:
    """Print what the removal did, or what a dry run would do."""
    prefix = '[DRY-RUN] Would ' if dry_run else ''
    lines: list[str] = []
    if not report.statements:
        lines.append('No row-level security object of this server is installed; nothing to remove.')
    else:
        count = f'{len(report.statements)} statement{"" if len(report.statements) == 1 else "s"}'
        lines.append(f'{prefix}issue {count}:' if dry_run else f'Issued {count}:')
        lines.extend(f'  {statement}' for statement in report.statements)
    if report.disabled:
        verb = 'disable' if dry_run else 'Disabled'
        lines.append(f'{prefix}{verb} row-level security on: {", ".join(report.disabled)}')
    if report.left_enabled:
        verb = 'leave' if dry_run else 'Left'
        tables = ', '.join(f'{table} ({reason})' for table, reason in report.left_enabled.items())
        lines.append(f'{prefix}{verb} row-level security enabled on: {tables}')
    print('\n'.join(lines), file=sys.stderr)


__all__ = ['run_disable_rls']
