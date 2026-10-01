"""Value types and limits shared by the embedding repository mixins and the tool layer.

Holds the ``ChunkEmbedding`` record written by the store paths, the
``MetadataFilterValidationError`` raised by both search paths, and the SQLite
``IN (...)`` batch size used by the bulk deletes and the compressed search.
"""

from dataclasses import dataclass

from app.errors import ControlFlowError

# SQLite caps host parameters per statement at SQLITE_MAX_VARIABLE_NUMBER (as low
# as 999 on older builds), so an unbounded candidate id list cannot be bound as a
# single IN (...) clause without risking 'too many SQL variables'. The compressed
# read batches the candidate ids in chunks below the most conservative limit;
# PostgreSQL is unaffected (it binds the whole set as one = ANY($1::uuid[]) array).
SQLITE_IN_CLAUSE_BATCH = 900


@dataclass
class ChunkEmbedding:
    """Embedding data for a single chunk with boundary information.

    This dataclass bundles the embedding vector with its character boundaries
    in the original document, enabling chunk-aware reranking. When embedding
    compression is enabled the optional ``payload`` field carries the
    provider-encoded bytes that the compressed write path persists to the
    ``vec_context_embeddings_compressed`` table; the fp32 write path ignores
    it.

    Attributes:
        embedding: The embedding vector for this chunk.
        start_index: Character offset where chunk starts in original document.
        end_index: Character offset where chunk ends in original document.
        payload: Optional compressed payload bytes produced by the active
            compression provider's ``encode_sync`` method. ``None`` when
            compression is disabled.

    Example:
        >>> chunk_emb = ChunkEmbedding(
        ...     embedding=[0.1, 0.2, 0.3],
        ...     start_index=0,
        ...     end_index=100
        ... )
    """

    embedding: list[float]
    start_index: int
    end_index: int
    payload: bytes | None = None


class MetadataFilterValidationError(ControlFlowError):
    """Exception raised when metadata filters fail validation.

    This exception enables unified error handling between search_context
    and semantic_search_context tools.

    Subclasses ``ControlFlowError`` -- mirroring its sibling
    :class:`~app.repositories.fts_repository.faults.FtsValidationError` -- because an
    invalid metadata filter is a client-input validation failure, normal control
    flow rather than a database fault. It is raised inside the read callables that
    run under ``get_connection`` (both backends, both compression modes), whose
    wrappers exempt ``ControlFlowError`` from circuit-breaker failure accounting;
    without this parentage a client repeatedly sending an invalid ``metadata_filters``
    to semantic/hybrid search would open the breaker into a process-wide outage.
    ``ControlFlowError`` subclasses ``Exception``, so every existing handler
    (including the tool-layer ``except MetadataFilterValidationError`` catches)
    still works unchanged.
    """

    def __init__(self, message: str, validation_errors: list[str]) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            validation_errors: List of validation error messages
        """
        super().__init__(message)
        self.message = message
        self.validation_errors = validation_errors
