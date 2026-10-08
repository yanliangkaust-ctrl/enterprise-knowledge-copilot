"""Small failure boundaries; no retries and no content-bearing exception text."""
from contextlib import contextmanager
from backend.observability import emit, record_metadata


class ServiceFailure(RuntimeError):
    def __init__(self, code, status=500):
        self.code, self.status = code, status
        super().__init__(code)


@contextmanager
def boundary(code, status=500):
    try:
        yield
    except ServiceFailure:
        raise
    except Exception as exc:
        emit("application_error", status="error", error_code=code,
             exception_type=type(exc).__name__, http_status=status)
        raise ServiceFailure(code, status) from None


def component(call, code):
    with boundary(code):
        return call()


def degraded(code, failed, final):
    record_metadata(degraded=True, failed_component=failed, final_route=final)
    emit("retrieval_degraded", reason_code=code, degraded=True,
         failed_component=failed, final_route=final)
