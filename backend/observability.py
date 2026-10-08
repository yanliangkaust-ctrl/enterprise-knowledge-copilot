"""Allowlisted JSON events and request context; never serialize RAG content."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import json
import logging
import math
import sys
from time import perf_counter
from uuid import uuid4

from starlette.responses import PlainTextResponse


@dataclass
class RequestContext:
    request_id: str
    metadata: dict = field(default_factory=dict)


request_context = ContextVar("request_observability", default=None)
current_stage = ContextVar("observability_stage", default="request")
ALLOWED_FIELDS = {
    "stage", "status", "duration_ms", "http_status", "reason_code", "error_code",
    "exception_type", "original_route", "final_route", "retriever", "requested_k",
    "effective_k", "candidate_count", "merged_candidate_count", "selected_count",
    "vector_candidate_count", "graph_candidate_count", "fallback_activated",
    "facet", "supported", "facet_count", "missing_facet_count", "knowledge_revision",
    "previous_revision", "new_revision", "document_count", "chunk_count",
    "embedding_backend", "relationship_support", "min_score", "max_score",
    "degraded", "failed_component", "reserved_count", "initialized", "index_available", "store_revision",
}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.event_payload, separators=(",", ":"), allow_nan=False)


logger = logging.getLogger("enterprise.observability")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(logging.NullHandler())


def configure_json_logging():
    """Enable stdout at application entry points, never on library import."""
    if any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def emit(event, **fields):
    context = request_context.get()
    payload = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(),
               "request_id": context.request_id if context else None,
               "stage": current_stage.get(), "status": "ok"}
    for key, value in fields.items():
        if key not in ALLOWED_FIELDS:
            continue
        if isinstance(value, (bool, int)) or value is None:
            payload[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            payload[key] = value
        elif isinstance(value, str):
            payload[key] = value[:160]
    logger.info("application_event", extra={"event_payload": payload})


def record_metadata(**fields):
    context = request_context.get()
    if context:
        context.metadata.update({k: v for k, v in fields.items() if k in ALLOWED_FIELDS})


@contextmanager
def stage(name):
    token = current_stage.set(name)
    started = perf_counter()
    status = "ok"
    try:
        yield
    except Exception as exc:
        status = "error"
        context = request_context.get()
        if context is None or "error_code" not in context.metadata:
            record_metadata(error_code="stage_failed", exception_type=type(exc).__name__, stage=name)
        if context is None:
            emit("application_error", stage=name, status="error", exception_type=type(exc).__name__, error_code="stage_failed")
        raise
    finally:
        emit("stage_completed", stage=name, status=status,
             duration_ms=max(0.0, (perf_counter() - started) * 1000))
        current_stage.reset(token)


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def decision(code, sufficient):
    record_metadata(reason_code=code)
    emit("sufficiency_check", stage="evidence_sufficiency", reason_code=code, supported=sufficient)


class RequestObservabilityMiddleware:
    """ASGI boundary covers validation and threadpool requests without body logging."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        context = RequestContext(str(uuid4()))
        token = request_context.set(context)
        started = perf_counter()
        response_started = False
        http_status = 500
        emit("request_started")

        async def observed_send(message):
            nonlocal response_started, http_status
            if message["type"] == "http.response.start":
                response_started = True
                http_status = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                message = {**message, "headers": [*headers, (b"x-request-id", context.request_id.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except Exception as exc:
            from backend.resilience import ServiceFailure
            failure_status = exc.status if isinstance(exc, ServiceFailure) else 500
            failure_code = exc.code if isinstance(exc, ServiceFailure) else "pipeline_internal_error"
            emit("application_error", stage=context.metadata.get("stage", "request"), status="error",
                 exception_type=type(exc).__name__, error_code=failure_code, http_status=failure_status)
            if response_started:
                raise RuntimeError("response_delivery_failed") from None
            # Generic existing server-failure response, without rethrowing a
            # content-bearing exception into Uvicorn's default traceback logger.
            await PlainTextResponse("Service Unavailable" if failure_status == 503 else "Internal Server Error", status_code=failure_status)(scope, receive, observed_send)
        finally:
            if http_status == 422:
                emit("application_error", status="error", error_code="request_validation_failed", http_status=422)
            safe_metadata = {k: v for k, v in context.metadata.items() if k not in {"stage", "error_code", "exception_type"}}
            emit("request_completed", **safe_metadata, stage="request",
                 status="error" if http_status >= 400 else "ok", http_status=http_status,
                 duration_ms=max(0.0, (perf_counter() - started) * 1000))
            request_context.reset(token)
