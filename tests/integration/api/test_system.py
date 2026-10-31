import pytest
from app.core.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI backend is running"
    assert response.json()["status"] == "ok"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.integration
def test_unknown_route_uses_standard_error_envelope(client):
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
    assert response.json()["error"]["request_id"]


@pytest.mark.integration
def test_production_disables_docs_and_requires_restricted_hosts():
    settings = Settings(
        environment="production",
        docs_enabled=True,
        allowed_hosts=["testserver"],
    )

    with TestClient(create_app(settings)) as production_client:
        assert production_client.get("/docs").status_code == 404
        assert production_client.get("/").status_code == 200
