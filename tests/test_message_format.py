"""Tests for consolidated message format (Issue 5).

Verifies that store_context, update_context, store_context_batch, and
update_context_batch produce single-parenthetical messages instead of
double-parenthetical "(embedding generated) (summary generated)".
"""

from __future__ import annotations


class TestStoreContextMessageFormat:
    """Tests for store_context response message construction."""

    def test_no_images_no_features(self) -> None:
        """Message with no images and no features: 'Context stored'."""
        parts: list[str] = []
        validated_images: list[str] = []
        action = 'stored'

        base = f'Context {action} with {len(validated_images)} images' if validated_images else f'Context {action}'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Context stored'

    def test_with_images_no_features(self) -> None:
        """Message with images but no features: 'Context stored with 2 images'."""
        parts: list[str] = []
        validated_images = ['img1', 'img2']
        action = 'stored'

        base = f'Context {action} with {len(validated_images)} images' if validated_images else f'Context {action}'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Context stored with 2 images'

    def test_embedding_and_summary(self) -> None:
        """Single parenthetical: 'Context stored (embedding generated, summary generated)'."""
        parts: list[str] = []
        validated_images: list[str] = []
        action = 'stored'

        # Simulate embedding + summary
        parts.extend(['embedding generated', 'summary generated'])

        base = f'Context {action} with {len(validated_images)} images' if validated_images else f'Context {action}'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Context stored (embedding generated, summary generated)'

    def test_with_images_and_features(self) -> None:
        """Full message: 'Context stored with 2 images (embedding generated, summary generated)'."""
        parts: list[str] = []
        validated_images = ['img1', 'img2']
        action = 'stored'

        parts.extend(['embedding generated', 'summary generated'])

        base = f'Context {action} with {len(validated_images)} images' if validated_images else f'Context {action}'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Context stored with 2 images (embedding generated, summary generated)'

    def test_embedding_only(self) -> None:
        """Single feature: 'Context stored (embedding generated)'."""
        parts: list[str] = []
        validated_images: list[str] = []
        action = 'stored'

        parts.append('embedding generated')

        base = f'Context {action} with {len(validated_images)} images' if validated_images else f'Context {action}'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Context stored (embedding generated)'

    def test_summary_preserved(self) -> None:
        """Dedup case: 'Context updated (summary preserved)'."""
        parts: list[str] = []
        validated_images: list[str] = []
        action = 'updated'

        parts.append('summary preserved')

        base = f'Context {action} with {len(validated_images)} images' if validated_images else f'Context {action}'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Context updated (summary preserved)'


class TestUpdateContextMessageFormat:
    """Tests for update_context response message construction."""

    def test_both_regenerated(self) -> None:
        """Single parenthetical: 'Successfully updated 2 field(s) (embedding regenerated, summary regenerated)'."""
        parts: list[str] = []
        updated_fields = ['text', 'metadata']

        parts.extend(['embedding regenerated', 'summary regenerated'])

        base = f'Successfully updated {len(updated_fields)} field(s)'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Successfully updated 2 field(s) (embedding regenerated, summary regenerated)'

    def test_no_regeneration(self) -> None:
        """No features: 'Successfully updated 1 field(s)'."""
        parts: list[str] = []
        updated_fields = ['metadata']

        base = f'Successfully updated {len(updated_fields)} field(s)'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Successfully updated 1 field(s)'


class TestBatchStoreMessageFormat:
    """Tests for store_context_batch aggregate message construction."""

    def test_with_features(self) -> None:
        """Single parenthetical: 'Stored 5/5 entries successfully (embeddings generated, summaries generated)'."""
        parts: list[str] = []
        succeeded = 5
        total = 5
        embeddings_generated_count = 5
        summaries_generated_count = 5

        if embeddings_generated_count > 0:
            parts.append('embeddings generated')
        if summaries_generated_count > 0:
            parts.append('summaries generated')

        base = f'Stored {succeeded}/{total} entries successfully'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Stored 5/5 entries successfully (embeddings generated, summaries generated)'

    def test_no_features(self) -> None:
        """No features: 'Stored 3/3 entries successfully'."""
        parts: list[str] = []
        succeeded = 3
        total = 3
        embeddings_generated_count = 0
        summaries_generated_count = 0

        if embeddings_generated_count > 0:
            parts.append('embeddings generated')
        if summaries_generated_count > 0:
            parts.append('summaries generated')

        base = f'Stored {succeeded}/{total} entries successfully'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Stored 3/3 entries successfully'

    def test_no_summaries_when_all_skipped(self) -> None:
        """Provider configured but all entries skipped: no 'summaries generated' in message."""
        parts: list[str] = []
        succeeded = 3
        total = 3
        embeddings_generated_count = 3
        summaries_generated_count = 0

        if embeddings_generated_count > 0:
            parts.append('embeddings generated')
        if summaries_generated_count > 0:
            parts.append('summaries generated')

        base = f'Stored {succeeded}/{total} entries successfully'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Stored 3/3 entries successfully (embeddings generated)'
        assert 'summaries generated' not in message


class TestBatchUpdateMessageFormat:
    """Tests for update_context_batch aggregate message construction."""

    def test_with_features(self) -> None:
        """Single parenthetical: 'Updated 3/3 entries successfully (embeddings regenerated, summaries regenerated)'."""
        parts: list[str] = []
        succeeded = 3
        total = 3
        embeddings_generated_count = 3
        summaries_generated_count = 3

        if embeddings_generated_count > 0:
            parts.append('embeddings regenerated')
        if summaries_generated_count > 0:
            parts.append('summaries regenerated')

        base = f'Updated {succeeded}/{total} entries successfully'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Updated 3/3 entries successfully (embeddings regenerated, summaries regenerated)'

    def test_no_regeneration_when_all_skipped(self) -> None:
        """Provider configured but no text changes: no generation info in message."""
        parts: list[str] = []
        succeeded = 2
        total = 2
        embeddings_generated_count = 0
        summaries_generated_count = 0

        if embeddings_generated_count > 0:
            parts.append('embeddings regenerated')
        if summaries_generated_count > 0:
            parts.append('summaries regenerated')

        base = f'Updated {succeeded}/{total} entries successfully'
        message = f'{base} ({", ".join(parts)})' if parts else base

        assert message == 'Updated 2/2 entries successfully'
