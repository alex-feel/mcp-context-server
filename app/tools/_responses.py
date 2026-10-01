"""Success-message text for single-entry and batch store and update responses."""


def build_store_response_message(
    *,
    action: str,
    image_count: int,
    embedding_generated: bool,
    embedding_stored: bool,
    summary_generated: bool,
    summary_preserved: bool,
) -> str:
    """Build a response message for a store operation.

    Constructs a human-readable message with parenthetical detail parts
    covering embedding status, summary status, and image count.

    Args:
        action: 'stored' or 'updated' (deduplication outcome)
        image_count: Number of validated images (0 suppresses image mention)
        embedding_generated: Whether embeddings were generated
        embedding_stored: Whether generated embeddings were stored to DB
        summary_generated: Whether a new summary was generated
        summary_preserved: Whether an existing summary was reused

    Returns:
        Formatted message string like 'Context stored (embedding generated, summary generated)'.
    """
    parts: list[str] = []

    if embedding_generated and not embedding_stored:
        parts.append('embedding generated but not stored - duplicate')
    elif embedding_stored:
        parts.append('embedding generated')

    if summary_generated:
        parts.append('summary generated')
    elif summary_preserved:
        parts.append('summary preserved')

    # Suppress "with 0 images" when no images
    base = f'Context {action} with {image_count} images' if image_count > 0 else f'Context {action}'

    # Single consolidated parenthetical
    return f'{base} ({", ".join(parts)})' if parts else base


def build_update_response_message(
    *,
    updated_fields_count: int,
    embedding_generated: bool,
    summary_generated: bool,
    summary_cleared: bool,
) -> str:
    """Build a response message for an update operation.

    Args:
        updated_fields_count: Number of fields updated
        embedding_generated: Whether embeddings were regenerated
        summary_generated: Whether summary was regenerated
        summary_cleared: Whether existing summary was cleared

    Returns:
        Formatted message string.
    """
    parts: list[str] = []
    if embedding_generated:
        parts.append('embedding regenerated')
    if summary_generated:
        parts.append('summary regenerated')
    elif summary_cleared:
        parts.append('summary cleared')

    base = f'Successfully updated {updated_fields_count} field(s)'
    return f'{base} ({", ".join(parts)})' if parts else base


def build_batch_store_response_message(
    *,
    succeeded: int,
    total: int,
    embeddings_generated_count: int,
    embeddings_stored_count: int,
    summaries_generated_count: int,
    summaries_preserved_count: int,
) -> str:
    """Build a response message for a batch store operation.

    Args:
        succeeded: Number of successfully stored entries
        total: Total number of entries in the batch
        embeddings_generated_count: Number of entries with generated embeddings
        embeddings_stored_count: Number of entries where embeddings were stored
        summaries_generated_count: Number of entries with generated summaries
        summaries_preserved_count: Number of entries with preserved summaries

    Returns:
        Formatted batch message string.
    """
    parts: list[str] = []
    if embeddings_generated_count > 0:
        not_stored = embeddings_generated_count - embeddings_stored_count
        if not_stored > 0:
            parts.append(f'embeddings generated ({not_stored} not stored - duplicates)')
        else:
            parts.append('embeddings generated')
    if summaries_generated_count > 0:
        parts.append('summaries generated')
    if summaries_preserved_count > 0:
        parts.append('summaries preserved')
    base = f'Stored {succeeded}/{total} entries successfully'
    return f'{base} ({", ".join(parts)})' if parts else base


def build_batch_update_response_message(
    *,
    succeeded: int,
    total: int,
    embeddings_generated_count: int,
    summaries_generated_count: int,
    summaries_cleared_count: int,
) -> str:
    """Build a response message for a batch update operation.

    Args:
        succeeded: Number of successfully updated entries
        total: Total number of entries in the batch
        embeddings_generated_count: Number of entries with regenerated embeddings
        summaries_generated_count: Number of entries with regenerated summaries
        summaries_cleared_count: Number of entries with cleared summaries

    Returns:
        Formatted batch message string.
    """
    parts: list[str] = []
    if embeddings_generated_count > 0:
        parts.append('embeddings regenerated')
    if summaries_generated_count > 0:
        parts.append('summaries regenerated')
    if summaries_cleared_count > 0:
        parts.append('summaries cleared')
    base = f'Updated {succeeded}/{total} entries successfully'
    return f'{base} ({", ".join(parts)})' if parts else base
