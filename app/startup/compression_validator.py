"""Startup validator for embedding compression seed and provenance.

When ``ENABLE_EMBEDDING_COMPRESSION`` is true the server reads the singleton
``compression_metadata`` table and reconciles it against the environment:

- Bootstrap mode (table empty): INSERT a provenance row from the current
  environment (``COMPRESSION_SEED`` defaults to 0) and become the source
  of truth.
- Validation mode (table populated): the env values MUST match the persisted
  row. ``provider``, ``bits``, ``variant``, ``seed`` and ``dim`` are compared
  on every start.

All mismatches raise :class:`ConfigurationError` (exit 78) so the supervisor
does not auto-restart.
"""

import logging
import sqlite3
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.compression.provenance import insert_compression_metadata
from app.compression.provenance import read_compression_metadata
from app.compression.types import CompressionMetadata
from app.errors import ConfigurationError
from app.settings import get_settings

logger = logging.getLogger(__name__)


async def validate_compression_provenance(backend: StorageBackend) -> None:
    """Bootstrap-or-validate compression provenance.

    Args:
        backend: Storage backend instance.

    Raises:
        ConfigurationError: When the env-derived compression configuration
            disagrees with the stored provenance row (exit 78). On the first
            start (no provenance row yet) the current configuration is recorded
            as the singleton row instead of raising; ``COMPRESSION_SEED`` always
            resolves (it defaults to 0) so it is never "missing" at bootstrap.
    """
    settings = get_settings()
    comp = settings.compression
    if not comp.enabled:
        return

    db_meta = await read_compression_metadata(backend)

    if db_meta is None:
        meta = CompressionMetadata(
            provider=comp.provider,
            bits=comp.bits,
            variant=comp.variant,
            seed=comp.seed,
            dim=settings.embedding.dim,
        )
        try:
            await insert_compression_metadata(backend, meta)
        except (asyncpg.UniqueViolationError, sqlite3.IntegrityError):
            # Two concurrent startup instances may both observe
            # ``db_meta is None`` and race the INSERT. The losing
            # instance's INSERT raises against the
            # ``compression_metadata CHECK (id = 1)`` singleton
            # constraint. Re-read the now-committed row and either
            # accept (env matches the inherited row) or raise
            # informatively (env disagrees on any field). The
            # inherited-row comparison mirrors the validation-mode
            # branch below.
            inherited = await read_compression_metadata(backend)
            if inherited is None:
                raise ConfigurationError(
                    'Compression provenance integrity violation '
                    'occurred during bootstrap but no provenance row '
                    'is visible after re-read. Check database '
                    'consistency and replication state before '
                    'restarting.',
                ) from None
            env_provider_race: str = cast(str, comp.provider)
            env_variant_race: str = cast(str, comp.variant)
            inherited_provider: str = cast(str, inherited.provider)
            inherited_variant: str = cast(str, inherited.variant)
            race_mismatches: list[str] = []
            if comp.seed != inherited.seed:
                race_mismatches.append(
                    f'COMPRESSION_SEED env={comp.seed} '
                    f'db={inherited.seed}',
                )
            if env_provider_race != inherited_provider:
                race_mismatches.append(
                    f'COMPRESSION_PROVIDER env={env_provider_race} '
                    f'db={inherited_provider}',
                )
            if comp.bits != inherited.bits:
                race_mismatches.append(
                    f'COMPRESSION_BITS env={comp.bits} '
                    f'db={inherited.bits}',
                )
            if env_variant_race != inherited_variant:
                race_mismatches.append(
                    f'COMPRESSION_VARIANT env={env_variant_race} '
                    f'db={inherited_variant}',
                )
            if settings.embedding.dim != inherited.dim:
                race_mismatches.append(
                    f'EMBEDDING_DIM env={settings.embedding.dim} '
                    f'db={inherited.dim}',
                )
            if race_mismatches:
                raise ConfigurationError(
                    'Compression provenance bootstrap race detected: '
                    'another startup instance committed a DIFFERENT '
                    'provenance row first. This instance must inherit '
                    "the winner's configuration. Restart with the env "
                    'values matching the persisted row. Mismatches: '
                    + '; '.join(race_mismatches),
                ) from None
            logger.info(
                'Compression provenance bootstrap race: another '
                'startup instance won; proceeding with inherited row: '
                'provider=%s bits=%d variant=%s dim=%d',
                inherited.provider,
                inherited.bits,
                inherited.variant,
                inherited.dim,
            )
            return
        logger.info(
            'Compression provenance bootstrapped: '
            'provider=%s bits=%d variant=%s dim=%d',
            meta.provider, meta.bits, meta.variant, meta.dim,
        )
        return

    # Cast the env-side Literal fields to ``str`` so the equality comparisons
    # below remain semantically meaningful for type checkers. Without the cast
    # mypy concludes the comparison is always False because both operands are
    # the same Literal type (currently both are ``Literal['turboquant']`` and
    # ``Literal['mse', 'ip']`` -- a static "no overlap" narrowing). The
    # comparison is still load-bearing at runtime because a corrupted DB row
    # (e.g. manual SQL tampering) could store a foreign value.
    env_provider: str = cast(str, comp.provider)
    env_variant: str = cast(str, comp.variant)
    db_provider: str = cast(str, db_meta.provider)
    db_variant: str = cast(str, db_meta.variant)

    mismatches: list[str] = []
    if comp.seed != db_meta.seed:
        mismatches.append(f'COMPRESSION_SEED env={comp.seed} db={db_meta.seed}')
    if env_provider != db_provider:
        mismatches.append(
            f'COMPRESSION_PROVIDER env={env_provider} db={db_provider}',
        )
    if comp.bits != db_meta.bits:
        mismatches.append(f'COMPRESSION_BITS env={comp.bits} db={db_meta.bits}')
    if env_variant != db_variant:
        mismatches.append(
            f'COMPRESSION_VARIANT env={env_variant} db={db_variant}',
        )
    if settings.embedding.dim != db_meta.dim:
        mismatches.append(
            f'EMBEDDING_DIM env={settings.embedding.dim} db={db_meta.dim}',
        )

    if mismatches:
        raise ConfigurationError(
            'Compression provenance mismatch detected. The DB is the source '
            'of truth after first startup; either unset the conflicting env '
            'vars or restore them to the original bootstrap values. '
            'Mismatches: ' + '; '.join(mismatches),
        )

    logger.debug(
        'Compression provenance validated: '
        'provider=%s bits=%d variant=%s dim=%d',
        db_meta.provider, db_meta.bits, db_meta.variant, db_meta.dim,
    )


__all__ = ['validate_compression_provenance']
