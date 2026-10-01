"""Tests for the store and update response message builders in app.tools._responses."""

from app.tools._responses import build_batch_store_response_message
from app.tools._responses import build_batch_update_response_message
from app.tools._responses import build_store_response_message
from app.tools._responses import build_update_response_message


class TestBuildStoreResponseMessage:
    """Test build_store_response_message shared function."""

    def test_basic_stored(self) -> None:
        """Basic store with no extras produces simple message."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert msg == 'Context stored'

    def test_stored_with_images(self) -> None:
        """Store with images includes image count."""
        msg = build_store_response_message(
            action='stored', image_count=3,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert msg == 'Context stored with 3 images'

    def test_embedding_generated_and_stored(self) -> None:
        """Embedding generated and stored shows 'embedding generated'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=True, embedding_stored=True,
            summary_generated=False, summary_preserved=False,
        )
        assert 'embedding generated' in msg
        assert 'not stored' not in msg

    def test_embedding_generated_not_stored(self) -> None:
        """Embedding generated but not stored shows duplicate message."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=True, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert 'embedding generated but not stored - duplicate' in msg

    def test_summary_generated(self) -> None:
        """Summary generated shows 'summary generated'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=True, summary_preserved=False,
        )
        assert 'summary generated' in msg

    def test_summary_preserved(self) -> None:
        """Summary preserved shows 'summary preserved'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=True,
        )
        assert 'summary preserved' in msg

    def test_all_parts(self) -> None:
        """All flags set produces message with all parts."""
        msg = build_store_response_message(
            action='stored', image_count=2,
            embedding_generated=True, embedding_stored=True,
            summary_generated=True, summary_preserved=False,
        )
        assert 'Context stored with 2 images' in msg
        assert 'embedding generated' in msg
        assert 'summary generated' in msg

    def test_no_parts(self) -> None:
        """No flags set produces no parenthetical."""
        msg = build_store_response_message(
            action='updated', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert msg == 'Context updated'
        assert '(' not in msg


class TestBuildUpdateResponseMessage:
    """Test build_update_response_message shared function."""

    def test_basic_update(self) -> None:
        """Basic update with no extras."""
        msg = build_update_response_message(
            updated_fields_count=3,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=False,
        )
        assert msg == 'Successfully updated 3 field(s)'

    def test_embedding_regenerated(self) -> None:
        """Embedding regenerated shows in message."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=True,
            summary_generated=False,
            summary_cleared=False,
        )
        assert 'embedding regenerated' in msg

    def test_summary_regenerated(self) -> None:
        """Summary regenerated shows in message."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=False,
            summary_generated=True,
            summary_cleared=False,
        )
        assert 'summary regenerated' in msg

    def test_summary_cleared(self) -> None:
        """Summary cleared shows in message."""
        msg = build_update_response_message(
            updated_fields_count=1,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=True,
        )
        assert 'summary cleared' in msg

    def test_all_parts(self) -> None:
        """All flags set produces all parts."""
        msg = build_update_response_message(
            updated_fields_count=5,
            embedding_generated=True,
            summary_generated=True,
            summary_cleared=False,
        )
        assert 'embedding regenerated' in msg
        assert 'summary regenerated' in msg
        assert '5 field(s)' in msg


class TestBuildBatchStoreResponseMessage:
    """Test build_batch_store_response_message shared function."""

    def test_basic_batch_message(self) -> None:
        """Basic batch store message."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=0, summaries_preserved_count=0,
        )
        assert msg == 'Stored 3/3 entries successfully'

    def test_with_embeddings_not_stored(self) -> None:
        """Batch with some embeddings not stored shows duplicate count."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=3, embeddings_stored_count=1,
            summaries_generated_count=0, summaries_preserved_count=0,
        )
        assert 'embeddings generated (2 not stored - duplicates)' in msg

    def test_with_summaries_preserved(self) -> None:
        """Batch with preserved summaries shows count."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=2, summaries_preserved_count=1,
        )
        assert 'summaries generated' in msg
        assert 'summaries preserved' in msg


class TestBuildBatchUpdateResponseMessage:
    """Test build_batch_update_response_message shared function."""

    def test_basic_batch_update(self) -> None:
        """Basic batch update message."""
        msg = build_batch_update_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=0,
        )
        assert msg == 'Updated 3/3 entries successfully'

    def test_with_summaries_cleared(self) -> None:
        """Batch with cleared summaries shows count."""
        msg = build_batch_update_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=2,
        )
        assert 'summaries cleared' in msg
