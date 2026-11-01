"""Single-organization API-key authentication and bounded local rate limiting."""

import hmac
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from time import monotonic

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings


class AccessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, settings: Settings) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.settings = settings
        self.windows: OrderedDict[str, tuple[float, int]] = OrderedDict()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if request.method == "OPTIONS" or not path.startswith(self.settings.api_prefix):
            return await call_next(request)
        if path in {
            f"{self.settings.api_prefix}/health/live",
            f"{self.settings.api_prefix}/health/ready",
        }:
            return await call_next(request)

        def reject(code: int, message: str) -> JSONResponse:
            return JSONResponse(
                status_code=code,
                content={
                    "error": {
                        "code": "access_denied",
                        "message": message,
                        "request_id": getattr(request.state, "request_id", None),
                    }
                },
                headers={"Retry-After": "60"} if code == 429 else {},
            )

        expected = self.settings.api_key
        supplied = request.headers.get("X-API-Key", "")
        if expected and not hmac.compare_digest(supplied.encode(), expected.encode()):
            return reject(401, "A valid X-API-Key is required.")
        # Do not trust caller-controlled X-Forwarded-For headers.
        identity = request.client.host if request.client else "unknown"
        now = monotonic()
        start, count = self.windows.pop(identity, (now, 0))
        if now - start >= 60:
            start, count = now, 0
        self.windows[identity] = (start, count + 1)
        if len(self.windows) > 10_000:
            self.windows.popitem(last=False)
        if count >= self.settings.rate_limit_per_minute:
            return reject(429, "Request rate limit exceeded.")
        return await call_next(request)
