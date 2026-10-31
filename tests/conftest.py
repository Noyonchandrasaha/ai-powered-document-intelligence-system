import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app() -> FastAPI:
    settings = Settings(environment="test", docs_enabled=False, allowed_hosts=["*"])
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI):
    with TestClient(app) as test_client:
        yield test_client
