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


def compression_byte_alignment_error(dim: int, bits: int, variant: str) -> str | None:
    """Return a guidance message if the compression config breaks byte-alignment.

    The compressed READ path combines per-row payloads via {MSE,IP}Payload.concat,
    which slice on byte boundaries and raise when (dim * effective_bits) is not a
    multiple of 8. effective_bits is COMPRESSION_BITS-1 for variant 'ip' (one bit
    reserved for the QJL sign) and COMPRESSION_BITS for 'mse'. The store/encode
    path writes single rows and never hits concat, so a misaligned dim lets stores
    (and the destructive CLI --compress) succeed yet makes every compressed search
    raise an opaque ValueError. Callers fail loudly up front instead.

    Args:
        dim: Embedding dimension (EMBEDDING_DIM).
        bits: Bits per coordinate (COMPRESSION_BITS).
        variant: Compression variant ('ip' or 'mse').

    Returns:
        A guidance message when the constraint is violated, else None.
    """
    effective_bits = bits - 1 if variant == 'ip' else bits
    if (dim * effective_bits) % 8 == 0:
        return None
    return (
        'Embedding compression byte-alignment violation: the compressed read '
        'path requires (EMBEDDING_DIM * effective_bits) to be a multiple of 8, '
        "where effective_bits is COMPRESSION_BITS - 1 for variant 'ip' (one bit "
        "reserved for the QJL sign) and COMPRESSION_BITS for 'mse'. Got "
        f'EMBEDDING_DIM={dim}, COMPRESSION_BITS={bits}, COMPRESSION_VARIANT={variant} '
        f'(effective_bits={effective_bits}, product={dim * effective_bits}). Pick '
        'an EMBEDDING_DIM (or COMPRESSION_BITS) that satisfies the constraint.'
    )


async def validate_compression_provenance(backend: StorageBackend) -> None:
    """Bootstrap-or-validate compression provenance.

    Args:
        backend: Storage backend instance.

    Raises:
        ConfigurationError: When the active compression configuration violates
            the compressed read path's byte-alignment invariant
            ((EMBEDDING_DIM * effective_bits) not a multiple of 8), or when the
            env-derived compression configuration disagrees with the stored
            provenance row (exit 78). On the first start (no provenance row yet)
            the current configuration is recorded as the singleton row instead of
            raising; ``COMPRESSION_SEED`` always resolves (it defaults to 0) so it
            is never "missing" at bootstrap.
    """
    settings = get_settings()
    comp = settings.compression
    if not comp.enabled:
        return

    # Byte-alignment invariant for the compressed READ path (see helper docstring):
    # a misaligned dim lets stores succeed but makes every compressed search raise.
    align_error = compression_byte_alignment_error(
        settings.embedding.dim, comp.bits, comp.variant,
    )
    if align_error is not None:
        raise ConfigurationError(align_error)

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


__all__ = ['compression_byte_alignment_error', 'validate_compression_provenance']
