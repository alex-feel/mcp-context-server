"""Row-shaping pieces shared by every migration direction: the tag-deduplicating
source query and the access-control backfill values.
"""


# Tags are a SET of labels per entry, and the target schema enforces that with a
# UNIQUE index on (context_entry_id, tag). A legacy source predating the write-path
# deduplication can hold the same label twice for one entry, which would abort the
# whole run on the second INSERT, so the copy collapses the duplicates instead of
# carrying them across. MIN(id) keeps the ordering deterministic and identical on
# both backends.
SELECT_DISTINCT_TAGS_SQL = (
    'SELECT MIN(id) AS id, context_entry_id, tag FROM tags '
    'GROUP BY context_entry_id, tag ORDER BY id ASC'
)


def access_backfill_values() -> tuple[str, str]:
    """Return the (owner_id, visibility) pair stamped onto migrated rows.

    Source databases predate the access-control columns, so every migrated row
    is backfilled fail-closed: visibility 'private' and owner the configured
    default principal -- the same values the server-side access-control
    migration's ``ADD COLUMN ... DEFAULT`` backfill applies on an in-place
    upgrade. Shared by all three copy paths (SQLite-to-SQLite, to-PostgreSQL,
    and PostgreSQL-to-SQLite) so they cannot drift.

    Returns:
        The ``(owner_id, visibility)`` values for migrated context entries.
    """
    from app.settings import get_settings

    return get_settings().access_control.default_principal, 'private'
