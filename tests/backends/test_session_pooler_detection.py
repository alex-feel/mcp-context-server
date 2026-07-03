"""Tests for Supabase Session Pooler (Supavisor) detection functionality."""

import logging

import pytest


class TestSessionModePoolerDetection:
    """Test session-mode pooler detection in PostgreSQLBackend."""

    def test_warns_for_session_pooler_with_high_pool_max(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WARN and flag detection for *.pooler.supabase.com:5432 with high pool_max."""
        from app.backends import postgresql_backend
        from app.backends.postgresql_backend import PostgreSQLBackend

        monkeypatch.setattr(
            postgresql_backend.settings.storage, 'postgresql_pool_max', 20, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:5432/postgres',
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is True
        assert any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_detects_session_pooler_from_libpq_keyvalue_dsn(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A libpq key-value DSN (no URL scheme) is parsed for host/port too.

        Regression: ``urlsplit`` yields an empty hostname for the
        ``host=... port=...`` spelling that asyncpg also accepts, so the advisory
        silently never fired even on a real Supabase Session Pooler set via a
        key-value POSTGRESQL_CONNECTION_STRING.
        """
        from app.backends import postgresql_backend
        from app.backends.postgresql_backend import PostgreSQLBackend

        monkeypatch.setattr(
            postgresql_backend.settings.storage, 'postgresql_pool_max', 20, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string=(
                'host=aws-0-us-east-1.pooler.supabase.com port=5432 '
                'user=u password=p dbname=postgres'
            ),
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is True
        assert any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_no_warn_for_session_pooler_within_cap(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Detected but no WARN when pool_max is within the default cap."""
        from app.backends import postgresql_backend
        from app.backends.postgresql_backend import PostgreSQLBackend

        monkeypatch.setattr(
            postgresql_backend.settings.storage, 'postgresql_pool_max', 10, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:5432/postgres',
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is True
        assert not any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_transaction_mode_port_not_flagged(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transaction-mode pooler (port 6543) is not a session pooler."""
        from app.backends import postgresql_backend
        from app.backends.postgresql_backend import PostgreSQLBackend

        monkeypatch.setattr(
            postgresql_backend.settings.storage, 'postgresql_pool_max', 50, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:6543/postgres',
        )

        caplog.set_level(logging.WARNING)
        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is False
        assert not any('MaxClientsInSessionMode' in r.message for r in caplog.records)

    def test_direct_connection_not_flagged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A direct (non-Supabase) host is never flagged as a session pooler."""
        from app.backends import postgresql_backend
        from app.backends.postgresql_backend import PostgreSQLBackend

        monkeypatch.setattr(
            postgresql_backend.settings.storage, 'postgresql_pool_max', 100, raising=False,
        )

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@localhost:5432/mcp_context',
        )

        backend._detect_session_mode_pooler()

        assert backend._session_mode_pooler is False

    def test_metrics_include_session_pooler_when_detected(self) -> None:
        """get_metrics() reports session_mode_pooler_detected after detection runs."""
        from unittest.mock import MagicMock

        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@aws-0-us-east-1.pooler.supabase.com:5432/postgres',
        )
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=20)
        backend._session_mode_pooler = True

        metrics = backend.get_metrics()

        assert metrics['session_mode_pooler_detected'] is True

    def test_metrics_omit_session_pooler_before_detection_runs(self) -> None:
        """get_metrics() omits the field if detection never ran."""
        from unittest.mock import MagicMock

        from app.backends.postgresql_backend import PostgreSQLBackend

        backend = PostgreSQLBackend(
            connection_string='postgresql://u:p@localhost:5432/mcp_context',
        )
        backend._pool = MagicMock()
        backend._pool.get_size = MagicMock(return_value=5)
        backend._pool.get_idle_size = MagicMock(return_value=3)
        backend._pool.get_min_size = MagicMock(return_value=2)
        backend._pool.get_max_size = MagicMock(return_value=10)
        # _session_mode_pooler attribute not set (detection never ran)

        metrics = backend.get_metrics()

        assert 'session_mode_pooler_detected' not in metrics
