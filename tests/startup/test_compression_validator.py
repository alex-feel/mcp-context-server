"""Tests for the compression-provenance startup validator.

Covers the three classes of behavior in ``validate_compression_provenance``:

- early-return when compression is disabled,
- bootstrap-INSERT when the singleton table is empty (requires seed),
- validation-mode env-vs-DB reconciliation (mismatches raise
  ``ConfigurationError`` with exit code 78).

Each test uses a fresh SQLite backend so the singleton row starts empty.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.errors import ConfigurationError
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings
from app.startup.compression_validator import validate_compression_provenance


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset ``get_settings`` cache before and after every test.

    Env-var monkeypatching for compression settings would otherwise leak
    into unrelated tests because the settings singleton is process-global.

    Yields:
        Control to the test body; both setup and teardown invalidate the
        cache.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with the standard schema pre-applied."""
    db_path = tmp_path / 'test_validator.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)


def _set_compression_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    seed: str | None = '42',
    bits: str = '4',
    variant: str = 'ip',
    provider: str = 'turboquant',
    dim: str = '1024',
) -> None:
    """Reset and configure the compression env vars + refresh the settings
    singleton cache so subsequent ``get_settings()`` returns the new values.
    """
    for var in (
        'ENABLE_EMBEDDING_COMPRESSION', 'COMPRESSION_SEED',
        'COMPRESSION_BITS', 'COMPRESSION_VARIANT', 'COMPRESSION_PROVIDER',
        'EMBEDDING_DIM',
    ):
        monkeypatch.delenv(var, raising=False)

    # Default is True (v3.0.0); explicit 'false' is required to disable so
    # the env state, not the settings default, drives the validator path.
    monkeypatch.setenv(
        'ENABLE_EMBEDDING_COMPRESSION', 'true' if enabled else 'false',
    )
    if seed is not None:
        monkeypatch.setenv('COMPRESSION_SEED', seed)
    monkeypatch.setenv('COMPRESSION_BITS', bits)
    monkeypatch.setenv('COMPRESSION_VARIANT', variant)
    monkeypatch.setenv('COMPRESSION_PROVIDER', provider)
    monkeypatch.setenv('EMBEDDING_DIM', dim)

    get_settings.cache_clear()
    # The migration module caches settings at import; refresh it so the
    # migration loader's enabled branch sees the new toggle.
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@pytest.mark.parametrize(
    ('variant', 'bits', 'dim'),
    [
        ('ip', '4', '1020'),   # ip effective_bits=3: 1020*3=3060, not a multiple of 8
        ('mse', '3', '1020'),  # mse effective_bits=3: 1020*3=3060, not a multiple of 8
    ],
)
@pytest.mark.asyncio
async def test_raises_on_byte_alignment_violation(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    bits: str,
    dim: str,
) -> None:
    """A (dim, variant, bits) that breaks the compressed read's byte-alignment
    invariant raises ConfigurationError at startup, before any provenance I/O."""
    _set_compression_env(monkeypatch, variant=variant, bits=bits, dim=dim)
    with pytest.raises(ConfigurationError, match='byte-alignment'):
        await validate_compression_provenance(backend=backend)


@pytest.mark.asyncio
async def test_byte_aligned_dim_passes_alignment_check(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default dim=1024 (multiple of 64) satisfies byte-alignment, so the
    validator proceeds to bootstrap without raising the alignment error."""
    _set_compression_env(monkeypatch, variant='ip', bits='4', dim='1024')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)


@pytest.mark.asyncio
async def test_returns_early_when_disabled(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When compression is disabled the validator returns silently and
    performs no DB I/O (the compression_metadata table does not exist
    because the migration was skipped)."""
    _set_compression_env(monkeypatch, enabled=False, seed=None)
    # Migration is skipped in disabled mode.
    await apply_compression_migration(backend=backend)

    await validate_compression_provenance(backend=backend)

    def _table_exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='compression_metadata'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_table_exists) is False


@pytest.mark.asyncio
async def test_raises_when_disabled_but_compressed_data_present(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling compression on a database that already holds a provenance row is
    refused at startup (exit 78): a bare env flip would route search to an empty
    fp32 table and silently lose the compressed embeddings. The legitimate
    --decompress flow clears the row first, so this fires only on a bare flip."""
    # Bootstrap a compressed database: enable, migrate, record the singleton row.
    _set_compression_env(monkeypatch, enabled=True, seed='42', bits='4', variant='ip', dim='1024')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    # Flip compression off by env var alone, leaving the provenance row in place.
    _set_compression_env(monkeypatch, enabled=False, seed=None)
    with pytest.raises(ConfigurationError, match='already holds compressed embeddings'):
        await validate_compression_provenance(backend=backend)


@pytest.mark.asyncio
async def test_bootstrap_inserts_provenance_row(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap mode INSERTs a singleton row from env when seed is set."""
    _set_compression_env(monkeypatch, seed='42', bits='4', variant='ip', dim='1024')
    await apply_compression_migration(backend=backend)

    await validate_compression_provenance(backend=backend)

    def _read(conn: sqlite3.Connection) -> tuple[str, int, str, int, int]:
        cur = conn.execute(
            'SELECT provider, bits, variant, seed, dim '
            'FROM compression_metadata WHERE id = 1',
        )
        row = cur.fetchone()
        assert row is not None
        return (row[0], int(row[1]), row[2], int(row[3]), int(row[4]))

    provider, bits, variant, seed, dim = await backend.execute_read(_read)
    assert provider == 'turboquant'
    assert bits == 4
    assert variant == 'ip'
    assert seed == 42
    assert dim == 1024


@pytest.mark.asyncio
async def test_validation_mode_env_matches_db(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation mode with env matching DB passes silently."""
    _set_compression_env(monkeypatch, seed='42')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)
    # Second start with identical env.
    await validate_compression_provenance(backend=backend)


@pytest.mark.asyncio
async def test_validation_mode_seed_mismatch_raises(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Differing env seed vs DB row raises ConfigurationError."""
    _set_compression_env(monkeypatch, seed='42')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    _set_compression_env(monkeypatch, seed='9999')
    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)
    msg = str(exc_info.value)
    assert 'COMPRESSION_SEED' in msg
    assert '9999' in msg
    assert '42' in msg


@pytest.mark.asyncio
async def test_validation_mode_bits_mismatch_raises(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Differing env bits vs DB row raises ConfigurationError."""
    _set_compression_env(monkeypatch, seed='42', bits='4')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    _set_compression_env(monkeypatch, seed='42', bits='2')
    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)
    assert 'COMPRESSION_BITS' in str(exc_info.value)


@pytest.mark.asyncio
async def test_validation_mode_variant_mismatch_raises(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Differing env variant vs DB row raises ConfigurationError."""
    _set_compression_env(monkeypatch, seed='42', variant='ip')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    _set_compression_env(monkeypatch, seed='42', variant='mse')
    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)
    assert 'COMPRESSION_VARIANT' in str(exc_info.value)


@pytest.mark.asyncio
async def test_validation_mode_dim_mismatch_raises(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Differing env embedding dim vs DB row raises ConfigurationError."""
    _set_compression_env(monkeypatch, seed='42', dim='1024')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    _set_compression_env(monkeypatch, seed='42', dim='2048')
    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)
    assert 'EMBEDDING_DIM' in str(exc_info.value)


@pytest.mark.asyncio
async def test_bootstrap_race_env_matches_db_logs_and_returns(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the INSERT races and the inherited row matches the env, the
    validator must log an informative message and return without raising.

    Simulates two concurrent startup instances: this instance attempts
    INSERT and loses to a concurrent one; the singleton ``CHECK (id=1)``
    constraint raises ``sqlite3.IntegrityError``; the validator re-reads
    the inherited row (matching env) and logs the race outcome.
    """
    _set_compression_env(
        monkeypatch, seed='42', bits='4', variant='ip',
        provider='turboquant', dim='1024',
    )
    await apply_compression_migration(backend=backend)

    # Pre-populate the singleton row to simulate the winning instance
    # having already committed.
    from app.compression.provenance import insert_compression_metadata
    from app.compression.types import CompressionMetadata
    winning_meta = CompressionMetadata(
        provider='turboquant', bits=4, variant='ip', seed=42, dim=1024,
    )
    await insert_compression_metadata(backend, winning_meta)

    # Hide the row from the validator's first read so it takes the
    # bootstrap branch, then expose it on the post-race re-read.
    from app.compression import provenance as provenance_module
    real_read = provenance_module.read_compression_metadata
    call_count = {'n': 0}

    async def _hide_then_reveal(b: StorageBackend) -> CompressionMetadata | None:
        call_count['n'] += 1
        if call_count['n'] == 1:
            return None  # Triggers bootstrap branch.
        return await real_read(b)  # Post-race re-read sees the row.

    monkeypatch.setattr(
        'app.startup.compression_validator.read_compression_metadata',
        _hide_then_reveal,
    )

    import logging as logging_module
    caplog.set_level(logging_module.INFO)
    await validate_compression_provenance(backend=backend)

    log_text = caplog.text
    assert 'bootstrap race' in log_text
    assert 'another startup instance won' in log_text
    assert 'provider=turboquant' in log_text


@pytest.mark.asyncio
async def test_bootstrap_race_env_disagrees_with_db_raises_configuration_error(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the INSERT races and the inherited row disagrees with env
    on any field, the validator must raise ConfigurationError with a
    field-by-field mismatch diff.
    """
    _set_compression_env(
        monkeypatch, seed='42', bits='4', variant='ip',
        provider='turboquant', dim='1024',
    )
    await apply_compression_migration(backend=backend)

    # Pre-populate with a DIFFERENT seed than the env value.
    from app.compression.provenance import insert_compression_metadata
    from app.compression.types import CompressionMetadata
    conflicting_meta = CompressionMetadata(
        provider='turboquant', bits=4, variant='ip', seed=9999, dim=1024,
    )
    await insert_compression_metadata(backend, conflicting_meta)

    from app.compression import provenance as provenance_module
    real_read = provenance_module.read_compression_metadata
    call_count = {'n': 0}

    async def _hide_then_reveal(b: StorageBackend) -> CompressionMetadata | None:
        call_count['n'] += 1
        if call_count['n'] == 1:
            return None
        return await real_read(b)

    monkeypatch.setattr(
        'app.startup.compression_validator.read_compression_metadata',
        _hide_then_reveal,
    )

    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)

    msg = str(exc_info.value)
    assert 'bootstrap race detected' in msg
    assert 'COMPRESSION_SEED env=42 db=9999' in msg
    assert ConfigurationError.EXIT_CODE == 78


@pytest.mark.asyncio
async def test_bootstrap_race_post_race_read_empty_raises(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the INSERT races but the post-race re-read finds no row,
    the validator must raise ConfigurationError pointing to deeper
    consistency / replication issues.
    """
    _set_compression_env(
        monkeypatch, seed='42', bits='4', variant='ip',
        provider='turboquant', dim='1024',
    )
    await apply_compression_migration(backend=backend)

    from app.compression.types import CompressionMetadata

    # Force both reads to return None (simulates replication lag where
    # the winning row is committed but not yet visible to this reader).
    async def _always_none(_b: StorageBackend) -> CompressionMetadata | None:
        return None

    monkeypatch.setattr(
        'app.startup.compression_validator.read_compression_metadata',
        _always_none,
    )

    # Force INSERT to raise IntegrityError as if the winning instance
    # had committed.
    async def _raise_integrity(
        _b: StorageBackend,
        _m: CompressionMetadata,
    ) -> None:
        raise sqlite3.IntegrityError(
            'UNIQUE constraint failed: compression_metadata.id',
        )

    monkeypatch.setattr(
        'app.startup.compression_validator.insert_compression_metadata',
        _raise_integrity,
    )

    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)

    msg = str(exc_info.value)
    assert 'integrity violation' in msg
    assert 'no provenance row is visible' in msg
    assert ConfigurationError.EXIT_CODE == 78


@pytest.mark.asyncio
async def test_bootstrap_records_codebook_fingerprint(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap records a non-null SHA-256 digest of the realized rotation matrix."""
    _set_compression_env(monkeypatch, seed='42', bits='4', variant='ip', dim='1024')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    from app.compression.provenance import read_compression_metadata

    meta = await read_compression_metadata(backend)
    assert meta is not None
    assert meta.codebook_fingerprint is not None
    assert len(meta.codebook_fingerprint) == 64
    int(meta.codebook_fingerprint, 16)  # valid lowercase hex, raises if not


@pytest.mark.asyncio
async def test_validation_codebook_fingerprint_mismatch_raises(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted fingerprint the realized rotation can no longer reproduce
    (simulating a cross-host BLAS/LAPACK/CPU QR divergence) raises
    ConfigurationError even though every scalar config field still matches."""
    _set_compression_env(monkeypatch, seed='42', bits='4', variant='ip', dim='1024')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    def _tamper(conn: sqlite3.Connection) -> None:
        conn.execute(
            'UPDATE compression_metadata SET codebook_fingerprint = ? WHERE id = 1',
            ('0' * 64,),
        )

    await backend.execute_write(_tamper)

    with pytest.raises(ConfigurationError) as exc_info:
        await validate_compression_provenance(backend=backend)
    assert 'fingerprint' in str(exc_info.value).lower()
    assert ConfigurationError.EXIT_CODE == 78


@pytest.mark.asyncio
async def test_validation_null_fingerprint_warns_not_raises(
    backend: StorageBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pre-fingerprint provenance row (NULL fingerprint) cannot be validated
    for cross-host divergence: the validator warns and proceeds, never raises."""
    _set_compression_env(monkeypatch, seed='42', bits='4', variant='ip', dim='1024')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    def _null_fingerprint(conn: sqlite3.Connection) -> None:
        conn.execute(
            'UPDATE compression_metadata SET codebook_fingerprint = NULL WHERE id = 1',
        )

    await backend.execute_write(_null_fingerprint)

    import logging as logging_module
    caplog.set_level(logging_module.WARNING)
    await validate_compression_provenance(backend=backend)  # must NOT raise
    assert 'fingerprint' in caplog.text.lower()


@pytest.mark.asyncio
async def test_generation_disabled_without_row_skips_seeding(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With embedding generation off and no provenance row the validator is a no-op.

    Embedding storage -- and with it the compression schema -- is provisioned
    from ENABLE_EMBEDDING_GENERATION, so nothing can ever write a compressed
    payload on this configuration. Seeding a provenance row here would wedge a
    later ENABLE_EMBEDDING_COMPRESSION=false flip behind the disable-direction
    guard's --decompress instruction on a deployment whose embedding
    infrastructure was never provisioned.
    """
    _set_compression_env(monkeypatch, enabled=True)
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    # The migration is gated the same way, so no compression tables exist.
    await apply_compression_migration(backend=backend)

    await validate_compression_provenance(backend=backend)

    def _table_exists(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='compression_metadata'",
        )
        return cur.fetchone() is not None

    assert await backend.execute_read(_table_exists) is False


@pytest.mark.asyncio
async def test_generation_disabled_with_existing_row_still_validates(
    backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provenance row seeded while generation was on is still validated.

    Data compressed earlier stays readable through the decode path, so the
    seed-locked invariant must keep protecting it after the operator turns
    embedding generation off: a seed mismatch still refuses startup (exit 78)
    rather than silently corrupting every decode.
    """
    _set_compression_env(monkeypatch, seed='42')
    await apply_compression_migration(backend=backend)
    await validate_compression_provenance(backend=backend)

    _set_compression_env(monkeypatch, seed='7')
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    with pytest.raises(ConfigurationError, match='COMPRESSION_SEED'):
        await validate_compression_provenance(backend=backend)
