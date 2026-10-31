import pytest
from app.api.v1 import health


@pytest.mark.integration
def test_liveness(client):
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


@pytest.mark.integration
def test_readiness(client):
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.integration
def test_readiness_returns_service_unavailable_when_database_is_down(
    client,
    monkeypatch: pytest.MonkeyPatch,
):
    async def database_is_down() -> bool:
        return False

    monkeypatch.setattr(health, "check_database_connection", database_is_down)

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "Database is not ready."
