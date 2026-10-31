# Production API

FastAPI assignment starter with typed settings, structured logging, request IDs, health probes, consistent errors, tests, and container support.

## Local setup

1. Create and activate a Python 3.12 virtual environment.
2. Install with: pip install -e ".[dev]"
3. Copy .env.example to .env.
4. Run: uvicorn app.main:app --reload

The API is available at http://localhost:8000.

Health checks:

- GET /api/v1/health/live
- GET /api/v1/health/ready

Interactive docs are at /docs while APP_DOCS_ENABLED=true.

The app adds common security headers and validates the Host header. In production, set
APP_ALLOWED_HOSTS to the real API hostnames and keep APP_DOCS_ENABLED=false.

## Quality checks

- pytest --cov=app --cov-report=term-missing --cov-fail-under=80
- ruff check .
- mypy app

Tests are organized under tests/unit for isolated logic and tests/integration for API-boundary behavior. Pytest's filesystem cache is disabled to avoid permission issues on shared Windows workspaces.

## Production

Set APP_ENVIRONMENT=production, APP_ALLOWED_HOSTS to your real hostnames, keep secrets outside the repository, restrict APP_ALLOWED_ORIGINS, and run behind a TLS-terminating reverse proxy or managed load balancer.
