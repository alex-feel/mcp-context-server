"""PostgreSQL target schema initialization and full-text search provisioning."""

from typing import TYPE_CHECKING

from app.cli.migrate_uuid.pg_connection import pg_column_exists
from app.cli.migrate_uuid.pg_connection import pg_connect
from app.cli.migrate_uuid.records import MigrationStats
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.pgvector_limits import exceeds_pgvector_index_dim_limit

if TYPE_CHECKING:
    import asyncpg


async def initialize_target_postgresql(
    target_url: str,
    *,
    embedding_dim: int | None,
    with_semantic: bool,
    source_has_fts: bool,
    stats: MigrationStats,
) -> None:
    """Auto-initialize a PostgreSQL target's v3 schema, mirroring
    :func:`initialize_target_sqlite`.

    Applies the base schema and the migrations the live server applies at
    startup, in the same order, against a backend built from ``target_url``:
    ``init_database`` -> (optional semantic search) -> jsonb_merge_patch ->
    function search_path -> (optional chunking). The embedding-compression
    migration and the seed-locked provenance validator are NEVER run here, so
    the target retains the fp32 layout; compression is a separate, explicit
    ``--compress`` step.

    The semantic and chunking migrations are invoked with ``force=True`` so the
    fp32 vector layout is created regardless of the CLI process's
    ``ENABLE_EMBEDDING_GENERATION`` value (the server-side gate), and
    ``apply_semantic_search_migration``
    receives the SOURCE-detected ``embedding_dim`` so the target vector column
    width matches the data being copied (mirrors the SQLite path's
    ``detect_source_embedding_dim`` -> ``initialize_target_sqlite`` flow).

    Args:
        target_url: asyncpg DSN for the target database.
        embedding_dim: SOURCE embedding dimension, templated into the semantic
            vector column. Ignored when ``with_semantic`` is False. When None on
            a ``with_semantic`` run (the source ``embedding_metadata`` table
            exists but is empty), this function resolves the same
            ``settings.embedding.dim`` fallback the semantic migration would have
            applied and uses that resolved value everywhere -- for the
            capacity pre-flight, for the migration call, and for the reported
            dimension -- so the report can never disagree with the built DDL.
        with_semantic: When True, also create the semantic-search and chunking
            layout (PG->PG migrations that copy embeddings). When False (a
            cross-backend migration that drops embeddings), only the base schema
            and the PostgreSQL helper functions are created; the server creates
            the vector layout later, at the operator's configured dimension,
            when re-embedding.
        source_has_fts: When True, provision the tsvector FTS column + GIN index
            on the target regardless of the CLI process's ENABLE_FTS setting,
            mirroring the SQLite CLI target's source-presence gate. When False
            (the source had no FTS) it is left unprovisioned.
        stats: Mutated to record an informational warning describing the
            auto-init.

    Raises:
        RuntimeError: If a ``with_semantic`` run's effective embedding dimension
            (``embedding_dim`` when known, else the ``settings.embedding.dim``
            fallback) exceeds the pgvector index cap (raised BEFORE any target
            DDL, so the target schema is never left partially initialized); if
            the target schema cannot be created; or -- on a ``with_semantic``
            run -- if the pgvector extension cannot be created (insufficient
            privileges on a managed service such as Supabase, or pgvector not
            installed on the host at all). A ``with_semantic=False`` run never
            touches pgvector, so it initializes cleanly on a pgvector-less host.
    """
    # fp32 capability pre-flight, BEFORE any connection or target DDL: pgvector
    # cannot build an HNSW index over vector columns wider than
    # PGVECTOR_INDEX_DIM_LIMIT dimensions, and the semantic-search migration
    # this function force-applies templates a dimension into vector(dim) and then
    # builds that index. Without this check the run dies mid-pipeline at CREATE
    # INDEX and leaves the target schema partially initialized (base tables
    # created, vector layout half-built).
    #
    # The dimension the DDL will ACTUALLY use is the source-detected value when
    # known, else apply_semantic_search_migration's own settings.embedding.dim
    # fallback (it resolves embedding_dim=None to settings.embedding.dim). A
    # source whose embedding_metadata table exists but is empty detects as None,
    # so validating only the source-detected value would leave the settings
    # fallback -- the value the DDL templates -- unchecked and crash mid-index.
    # Resolve the same fallback here and validate whichever value the DDL uses.
    #
    # effective_dim is resolved ONCE here and then used for the pre-flight, for the
    # dimension passed to the semantic migration, and for the final report line, so
    # the reported dimension can never disagree with the dimension the DDL built.
    effective_dim: int | None = None
    if with_semantic:
        effective_dim = embedding_dim
        if effective_dim is None:
            from app.settings import get_settings

            effective_dim = get_settings().embedding.dim
        if exceeds_pgvector_index_dim_limit(effective_dim):
            source_clause = (
                f'source embedding dimension ({effective_dim})'
                if embedding_dim is not None
                else (
                    f'configured EMBEDDING_DIM ({effective_dim}, the fallback used because '
                    'the source embedding_metadata table is empty)'
                )
            )
            raise RuntimeError(
                f'Cannot initialize the target database: the {source_clause} exceeds the '
                f'pgvector index limit of {PGVECTOR_INDEX_DIM_LIMIT} dimensions for fp32 '
                f'vectors, so building the fp32 vector layout would fail at the HNSW '
                f'CREATE INDEX and leave the target schema partially initialized. '
                f'Embeddings of this dimension cannot be copied into an fp32 target. '
                f'Recovery: migrate from a source copy whose embedding tables '
                f'(embedding_metadata, vec_context_embeddings) are dropped so the target '
                f'initializes without the fp32 vector layout, then start the target server '
                f'with ENABLE_EMBEDDING_COMPRESSION=true (compressed payloads have no '
                f'pgvector dimension cap) and re-embed with --embed-missing.',
            )

    import asyncpg

    from app.backends import create_backend
    from app.backends.postgresql_backend.session import quote_pg_identifier
    from app.migrations.chunking import apply_chunking_migration
    from app.migrations.fts import apply_fts_migration
    from app.migrations.index_tree import apply_index_tree_migration
    from app.migrations.semantic import apply_function_search_path_migration
    from app.migrations.semantic import apply_jsonb_merge_patch_migration
    from app.migrations.semantic import apply_semantic_search_migration
    from app.settings import get_settings
    from app.startup import init_database

    schema = get_settings().storage.postgresql_schema

    # The target schema must exist before the schema-qualified function DDL in
    # the base schema / migrations runs (CREATE FUNCTION "<schema>".update_...);
    # PostgreSQL does not auto-create a non-default schema, so it is created
    # first and UNCONDITIONALLY. The pgvector extension is needed ONLY when the
    # fp32 vector layout will be built (with_semantic): a cross-backend
    # migration drops embeddings and must initialize cleanly on a pgvector-less
    # host (the pgvector-free compressed deployment shape), so it never issues
    # CREATE EXTENSION at all. When the extension IS needed it must exist
    # before the semantic migration's vector(dim) DDL and the backend pool's
    # vector-codec registration. Surface a clear, actionable error on managed
    # services where DDL privileges are restricted and on hosts where the
    # pgvector extension is not installed at all (missing control file --
    # IF NOT EXISTS does not suppress that failure).
    ext_conn = await pg_connect(target_url)
    try:
        try:
            await ext_conn.execute(f'CREATE SCHEMA IF NOT EXISTS {quote_pg_identifier(schema)}')
        except asyncpg.InsufficientPrivilegeError as exc:
            raise RuntimeError(
                'Cannot initialize the target database (insufficient privileges '
                f'to CREATE SCHEMA "{schema}"). Create the schema first, then '
                f'rerun: execute \'CREATE SCHEMA "{schema}";\' as a privileged user.',
            ) from exc
        if with_semantic:
            try:
                await ext_conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            except asyncpg.InsufficientPrivilegeError as exc:
                raise RuntimeError(
                    'Cannot initialize the target database (insufficient '
                    'privileges to CREATE EXTENSION vector). This migration '
                    'copies embeddings, which require pgvector. Enable it first, '
                    'then rerun: on Supabase use Dashboard -> Database -> '
                    'Extensions -> vector; on self-hosted PostgreSQL run '
                    '"CREATE EXTENSION vector;" as a superuser.',
                ) from exc
            except asyncpg.UndefinedFileError as exc:
                raise RuntimeError(
                    'Cannot initialize the target database: the pgvector '
                    'extension is not installed on the target PostgreSQL host. '
                    'This migration copies embeddings, which require pgvector. '
                    'Install it on the host first (for example use a '
                    'pgvector/pgvector image or the PostgreSQL pgvector package), '
                    'then rerun.',
                ) from exc
    finally:
        await ext_conn.close()

    # provision_vector mirrors with_semantic: a vector-carrying target has its
    # extension guaranteed by the block above (created or failed loudly), while
    # a vector-free target must not let the CLI process's env-driven gate force
    # pgvector provisioning it cannot satisfy on a pgvector-less host.
    backend = create_backend(
        backend_type='postgresql',
        connection_string=target_url,
        provision_vector=with_semantic,
    )
    await backend.initialize()
    try:
        await init_database(backend=backend)
        if with_semantic:
            await apply_semantic_search_migration(backend, force=True, embedding_dim=effective_dim)
        await apply_jsonb_merge_patch_migration(backend)
        await apply_function_search_path_migration(backend)
        # FTS: create the tsvector GENERATED column + GIN index ONLY when the
        # SOURCE had full-text search, mirroring the SQLite CLI target's
        # source-presence gate (initialize_target_sqlite keys FTS on
        # optional_tables['context_entries_fts']). force=True bypasses the CLI
        # process's ENABLE_FTS toggle so the migrated target's FTS capability is
        # decided solely by the source -- otherwise a SQLite->PG or PG->PG
        # migration run with ENABLE_FTS=false would silently drop the FTS the
        # SQLite target keeps. It MUST run BEFORE the data copy so the STORED
        # generated column auto-populates as rows are INSERTed.
        if source_has_fts:
            await apply_fts_migration(backend, force=True)
        if with_semantic:
            await apply_chunking_migration(backend, force=True)
        # index_tree node-summary table: provisioned regardless of with_semantic
        # (it concerns node summaries, not vectors) so a migrated target matches a
        # server-initialized DB. force=True mirrors the SQLite path; the table is
        # harmless when the node-summary feature is later disabled.
        await apply_index_tree_migration(backend, force=True)
    finally:
        await backend.shutdown()

    # Report the RESOLVED dimension the vector column was actually built at, not the
    # raw parameter: a source whose embedding_metadata table is empty detects as
    # None and the semantic migration falls back to settings.embedding.dim, so
    # printing the parameter would tell the operator that an irreversible schema
    # decision was made at an unknown width.
    stats.warnings.append(
        'auto-initialized target PostgreSQL schema '
        f'(semantic_search={"yes" if with_semantic else "no"}, '
        f'embedding_dim={effective_dim if with_semantic else "n/a"})',
    )


async def ensure_target_pg_fts(
    target_url: str,
    target_conn: 'asyncpg.Connection[asyncpg.Record]',
    *,
    target_schema: str,
    source_has_fts: bool,
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """Provision FTS on a PRE-EXISTING PostgreSQL target when the source has it.

    :func:`initialize_target_postgresql` provisions FTS only when it runs --
    that is, only when the target had NO ``context_entries`` table at all. A
    pre-existing target (its schema bootstrapped by a server started with
    ``ENABLE_FTS=false``, or created by any means other than this CLI)
    would silently lose the full-text search the source has -- the failure
    the source-presence gate prevents for freshly initialized targets.
    Unlike the embeddings backstop, which must ABORT (vectors are not
    derivable from the copied rows), FTS is fully derivable: the migration
    adds a STORED generated ``text_search_vector`` column plus its GIN index,
    so the backstop PROVISIONS it instead. It MUST run before the data copy
    so the generated column populates as rows are INSERTed. Mirrors the
    SQLite target path, which re-applies the IF-NOT-EXISTS FTS DDL keyed only
    on source presence.

    Args:
        target_url: asyncpg DSN for the target database.
        target_conn: Open target connection used for the column probe.
        target_schema: Explicit schema for the probe, matching the
            ``context_entries`` probe that established the target as
            pre-existing.
        source_has_fts: Whether the source carries full-text search.
        dry_run: When True, record the plan instead of provisioning.
        stats: Mutated with the provisioning (or plan) note.
    """
    if not source_has_fts:
        return
    if await pg_column_exists(
        target_conn, 'context_entries', 'text_search_vector', schema=target_schema,
    ):
        return
    if dry_run:
        stats.warnings.append(
            'source has full-text search but the pre-existing target lacks the '
            'text_search_vector column; it would be provisioned on a real run',
        )
        return

    from app.backends import create_backend
    from app.migrations.fts import apply_fts_migration

    # provision_vector=False: this backend runs FTS DDL only (tsvector column +
    # GIN index), which never touches the vector type, and the pre-existing
    # target may be a pgvector-less host (a compressed deployment) -- the CLI
    # process's env-driven gate must not force pgvector provisioning here.
    backend = create_backend(
        backend_type='postgresql',
        connection_string=target_url,
        provision_vector=False,
    )
    await backend.initialize()
    try:
        await apply_fts_migration(backend, force=True)
    finally:
        await backend.shutdown()
    stats.warnings.append(
        'provisioned full-text search on the pre-existing target '
        '(source has FTS; the target lacked the text_search_vector column)',
    )
