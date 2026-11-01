# Automated Test Results

Run date: 2026-11-01  
Environment: Windows, Python 3.13.9, local virtual environment

| Check | Result |
|---|---:|
| Pytest | 254 passed, 0 failed |
| Total `app` coverage | 81.04% (minimum: 80%) |
| Ruff (`app tests scripts`) | Passed |
| Mypy (`app`) | Passed, 43 source files |

Commands:

```powershell
python -m pytest --cov=app --cov-report=term --cov-fail-under=80
python -m ruff check app tests scripts
python -m mypy app
```

Tests use synthetic documents and controlled MongoDB/OpenAI adapters. They do
not prove live Google Vision/OpenAI service behavior or extraction accuracy on
real invoices. See [`evaluation.md`](evaluation.md) for the parser benchmark
and its limitations.
