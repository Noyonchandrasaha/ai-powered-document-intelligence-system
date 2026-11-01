import shutil
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from app.api.v1 import health
from app.core.config import Settings
from app.features.document_extraction.router import get_processing_service
from app.main import create_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


class NoOpProcessingService:
    async def enqueue_document(self, document_id: str) -> None:
        return None

    async def process_document(self, document_id: str) -> None:
        return None


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    settings = Settings(environment="test", docs_enabled=False, allowed_hosts=["*"])
    monkeypatch.setattr(
        health,
        "check_database_connection",
        lambda: _ready_database(),
    )
    application = create_app(settings)
    application.dependency_overrides[get_processing_service] = lambda: NoOpProcessingService()
    yield application
    application.dependency_overrides.pop(get_processing_service, None)


async def _ready_database() -> bool:
    return True


@pytest.fixture
def client(app: FastAPI):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    path = Path.cwd() / f".test-temp-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
