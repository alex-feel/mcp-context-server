"""The session's PostgreSQL test container belongs to this checkout's compose project."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.integration.postgresql.conftest import SESSION_CONTAINER
from tests.integration.postgresql.conftest import _project_name_for

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_the_session_container_runs_under_the_checkout_project(pg_test_url: str) -> None:
    project = _project_name_for(REPOSITORY_ROOT)
    port = urlsplit(pg_test_url).port

    result = subprocess.run(
        [
            'docker', 'ps',
            '--filter', f'label=com.docker.compose.project={project}',
            '--format', '{{.Ports}}',
        ],
        capture_output=True, text=True, check=True, timeout=30,
    )

    assert f':{port}->5432/tcp' in result.stdout


@pytest.mark.usefixtures('pg_test_url')
def test_the_session_records_that_it_brought_the_container_up() -> None:
    assert SESSION_CONTAINER.brought_up
