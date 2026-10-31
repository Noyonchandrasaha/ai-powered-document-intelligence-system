# Production API

Production-ready FastAPI starter with typed settings, structured logging, request IDs, health probes, consistent errors, tests, and container/CI support.

## Local setup

1. Create and activate a Python 3.12 virtual environment.
2. Install with: pip install -e ".[dev]"
3. Copy .env.example to .env.
4. Run: uvicorn app.main:app --reload

The API is available at http://localhost:8000.

Health checks:

- GET /api/v1/health/live
- GET /api/v1/health/ready

Interactive docs are at /docs while DOCS_ENABLED=true.

## Quality checks

- pytest
- ruff check .
- mypy app

## Production

Set APP_ENVIRONMENT=production, keep secrets outside the repository, restrict APP_ALLOWED_ORIGINS, and run behind a TLS-terminating reverse proxy or managed load balancer.
