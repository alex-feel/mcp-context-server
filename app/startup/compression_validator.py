"""Startup validator for embedding compression seed and provenance.

When ``ENABLE_EMBEDDING_COMPRESSION`` is true the server reads the singleton
``compression_metadata`` table and reconciles it against the environment:

- Bootstrap mode (table empty): INSERT a provenance row from the current
  environment (``COMPRESSION_SEED`` defaults to 0) and become the source
  of truth.
- Validation mode (table populated): the env values MUST match the persisted
  row. ``provider``, ``bits``, ``variant``, ``seed`` and ``dim`` are compared
  on every start. When those scalars match, the REALIZED codebook fingerprint
  (a SHA-256 of the ``numpy.linalg.qr`` rotation matrix) is re-derived and
  compared too, so a cross-host BLAS/LAPACK/CPU QR divergence -- the same
  ``(dim, seed)`` materializing a different rotation -- is caught before it can
  silently corrupt every decode/search.

All mismatches raise :class:`ConfigurationError` (exit 78) so the supervisor
does not auto-restart.
"""

import asyncio
import logging
import sqlite3
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.compression.provenance import insert_compression_metadata
from app.compression.provenance import read_compression_metadata
from app.compression.types import CompressionMetadata
from app.errors import ConfigurationError
from app.migrations.compression import uncompressed_fp32_guard_message
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


async def _compute_codebook_fingerprint() -> str:
    """Compute the realized codebook fingerprint for the active compression config.

    Builds the configured compression provider and materializes its rotation matrix
    OFF the event loop (a QR factorization). ``get_cached_rotation`` memoizes by
    ``(dim, seed)``, so this shares the matrix the encode/decode path uses rather than
    building a second one.

    Returns:
        Lowercase hex SHA-256 digest of the realized codebook.
    """
    from app.compression.factory import create_compression_provider

    provider = create_compression_provider()
    return await asyncio.to_thread(provider.codebook_fingerprint)


def _codebook_fingerprint_mismatch_error(
    stored_fingerprint: str,
    realized_fingerprint: str,
) -> ConfigurationError:
    """Build the ConfigurationError raised when the realized codebook diverges.

    Args:
        stored_fingerprint: The fingerprint recorded when the data was first compressed.
        realized_fingerprint: The fingerprint this host materializes now.

    Returns:
        A :class:`ConfigurationError` describing the divergence (exit 78).
    """
    return ConfigurationError(
        'Compression codebook fingerprint mismatch. The realized numpy.linalg.qr '
        'rotation matrix for this (dim, seed) does NOT match the one recorded when the '
        'compressed data was first written -- typically a different BLAS/LAPACK build '
        'or CPU (numpy.linalg.qr is not bit-reproducible across hosts even for a fixed '
        'seed). Reading compressed embeddings with a divergent codebook silently '
        'corrupts every decode and inner-product estimate. Run on a host whose numerical '
        'libraries reproduce the original codebook, or restore from backup. '
        f'Expected fingerprint={stored_fingerprint}, realized={realized_fingerprint}.',
    )


async def validate_compression_provenance(backend: StorageBackend) -> None:
    """Bootstrap-or-validate compression provenance.

    Args:
        backend: Storage backend instance.

    Raises:
        ConfigurationError: When the active compression configuration violates
            the compressed read path's byte-alignment invariant
            ((EMBEDDING_DIM * effective_bits) not a multiple of 8), when the
            env-derived compression configuration disagrees with the stored
            provenance row, or when the realized codebook fingerprint (the
            ``numpy.linalg.qr`` rotation matrix digest) diverges from the one
            recorded at first compression -- a cross-host BLAS/LAPACK/CPU QR
            divergence (exit 78). On the first start (no provenance row yet) the
            current configuration AND its realized fingerprint are recorded as the
            singleton row instead of raising; ``COMPRESSION_SEED`` always resolves
            (it defaults to 0) so it is never "missing" at bootstrap.
    """
    settings = get_settings()
    comp = settings.compression
    if not comp.enabled:
        # Compression is disabled. A FRESH/never-compressed database (no
        # provenance row) is fine -- return and let the server provision fp32
        # storage. But a database that ALREADY holds compressed data (a
        # provenance row is present) cannot be served as fp32 by a bare env
        # flip: the store/search/delete dispatch keys on this same runtime
        # toggle, so every read would route to the EMPTY recreated fp32 table
        # and silently return nothing. Refuse loudly (exit 78) and direct the
        # operator to decode the data back to fp32 first. The legitimate
        # --decompress CLI clears the provenance row via an inline DELETE inside
        # its single atomic transaction (app/cli/migrate_compression.py), so a
        # post-decompress startup reads None here and proceeds normally.
        existing = await read_compression_metadata(backend)
        if existing is not None:
            raise ConfigurationError(
                'ENABLE_EMBEDDING_COMPRESSION is false but this database already '
                'holds compressed embeddings (a compression_metadata provenance row '
                'is present). Disabling compression by environment variable alone '
                'would route semantic and hybrid search to an empty fp32 table and '
                'silently return no results. Run "mcp-context-server-migrate '
                '--decompress" to decode the embeddings back to fp32 storage (which '
                'also clears the provenance row) BEFORE disabling compression, or '
                're-enable ENABLE_EMBEDDING_COMPRESSION to keep serving the '
                'compressed data.',
            )
        return

    # Byte-alignment invariant for the compressed READ path (see helper docstring):
    # a misaligned dim lets stores succeed but makes every compressed search raise.
    align_error = compression_byte_alignment_error(
        settings.embedding.dim, comp.bits, comp.variant,
    )
    if align_error is not None:
        raise ConfigurationError(align_error)

    db_meta = await read_compression_metadata(backend)

    if db_meta is None and not settings.embedding.generation_enabled:
        # Mirror the migration's enable-direction guard before skipping: a
        # populated, never-compressed fp32 store must refuse loudly (exit 78)
        # even while embedding generation is toggled off, otherwise a bare
        # compression flip on an archive/read-only deployment would boot
        # silently with every stored embedding invisible to search. Server
        # startup already raises in apply_compression_migration before
        # reaching this validator; the re-check keeps the validator safe when
        # invoked standalone.
        guard_message = await uncompressed_fp32_guard_message(backend)
        if guard_message is not None:
            raise ConfigurationError(guard_message)
        # Embedding storage -- and with it the compression schema -- is
        # provisioned from ENABLE_EMBEDDING_GENERATION, so with generation off
        # and no provenance row there is nothing to validate and nothing to
        # seed (on a fresh database without embedding infrastructure the
        # compression migration skips schema creation too). Seeding a row here
        # would wedge a later ENABLE_EMBEDDING_COMPRESSION=false flip behind
        # the disable-direction guard's --decompress instruction on a
        # deployment whose embedding_chunks / vector infrastructure was never
        # provisioned. A database that compressed data while generation was on
        # still carries its row and is validated below.
        logger.debug(
            'Compression provenance: skipped (embedding generation disabled, '
            'no provenance row)',
        )
        return

    if db_meta is None:
        meta = CompressionMetadata(
            provider=comp.provider,
            bits=comp.bits,
            variant=comp.variant,
            seed=comp.seed,
            dim=settings.embedding.dim,
            # Record the REALIZED rotation-matrix digest so later starts can detect a
            # cross-host BLAS/LAPACK/CPU QR divergence and fail loudly (exit 78) rather
            # than silently corrupting every decode/search.
            codebook_fingerprint=await _compute_codebook_fingerprint(),
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
            # The scalar config matches the winner, but if THIS host materializes a
            # different rotation matrix than the winner recorded, reads would corrupt.
            # meta.codebook_fingerprint is this instance's realized digest (computed
            # above for the INSERT we lost); compare it to the winner's persisted digest.
            if (
                inherited.codebook_fingerprint is not None
                and meta.codebook_fingerprint != inherited.codebook_fingerprint
            ):
                race_fingerprint_error = _codebook_fingerprint_mismatch_error(
                    inherited.codebook_fingerprint,
                    cast(str, meta.codebook_fingerprint),
                )
                raise race_fingerprint_error from None
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

    # Scalars match. Verify the REALIZED codebook (the numpy.linalg.qr rotation matrix)
    # equals the one recorded at bootstrap: the same (dim, seed) can materialize a
    # DIFFERENT rotation on a host with a different BLAS/LAPACK build or CPU dispatch,
    # which would silently corrupt every decode/inner-product estimate. Catch it loudly.
    if db_meta.codebook_fingerprint is None:
        logger.warning(
            'Compression provenance row predates codebook fingerprinting; a cross-host '
            'rotation-matrix divergence cannot be detected for this database. Re-compress '
            'from a backup on the target host to record a fingerprint.',
        )
    else:
        realized_fingerprint = await _compute_codebook_fingerprint()
        if realized_fingerprint != db_meta.codebook_fingerprint:
            fingerprint_error = _codebook_fingerprint_mismatch_error(
                db_meta.codebook_fingerprint, realized_fingerprint,
            )
            raise fingerprint_error

    logger.debug(
        'Compression provenance validated: '
        'provider=%s bits=%d variant=%s dim=%d',
        db_meta.provider, db_meta.bits, db_meta.variant, db_meta.dim,
    )


__all__ = ['compression_byte_alignment_error', 'validate_compression_provenance']
