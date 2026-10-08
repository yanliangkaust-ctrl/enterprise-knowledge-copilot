"""Bounded process-local fixed-window limit, keyed by the ASGI client address."""
from math import ceil
from threading import Lock
from time import monotonic
from starlette.responses import JSONResponse
from backend.observability import emit


class ClientRateLimiter:
    def __init__(self, limit=30, window_seconds=60, max_clients=1024, clock=monotonic):
        if limit <= 0 or window_seconds <= 0 or max_clients <= 0:
            raise ValueError("Rate limits must be positive")
        self.limit, self.window, self.capacity, self.clock = limit, window_seconds, max_clients, clock
        self._clients = {}
        self._lock = Lock()

    def check(self, client):
        with self._lock:
            now = self.clock()
            self._clients = {key: value for key, value in self._clients.items() if now - value[0] < self.window}
            if client not in self._clients:
                if len(self._clients) >= self.capacity:
                    return False, max(1, ceil(min(v[0] + self.window - now for v in self._clients.values())))
                self._clients[client] = (now, 0)
            started, count = self._clients[client]
            if count >= self.limit:
                return False, max(1, ceil(started + self.window - now))
            self._clients[client] = (started, count + 1)
            return True, 0


class QueryRateLimitMiddleware:
    def __init__(self, app, limiter):
        self.app, self.limiter = app, limiter

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and scope['method'] == 'POST' and scope['path'].rstrip('/') == '/query':
            allowed, retry = self.limiter.check((scope.get('client') or ('unknown',))[0])
            if not allowed:
                emit('application_error', status='error', error_code='query_rate_limited', http_status=429)
                return await JSONResponse({'detail': 'Query rate limit exceeded', 'reason_code': 'query_rate_limited'},
                                          status_code=429, headers={'Retry-After': str(retry)})(scope, receive, send)
        await self.app(scope, receive, send)
