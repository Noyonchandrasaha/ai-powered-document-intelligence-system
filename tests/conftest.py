import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    settings = Settings(environment="test", docs_enabled=False)
    return TestClient(create_app(settings))
