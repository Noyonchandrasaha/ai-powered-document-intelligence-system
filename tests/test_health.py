def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI backend is running"
    assert response.json()["status"] == "ok"


def test_liveness(client):
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_readiness(client):
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_unknown_route_uses_framework_error(client):
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
    assert response.json()["error"]["request_id"]
