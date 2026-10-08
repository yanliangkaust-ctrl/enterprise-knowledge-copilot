import pytest
from backend import api


@pytest.fixture(autouse=True)
def isolated_client_limit(monkeypatch):
    # Each test represents a fresh service instance; retain actual production limits.
    monkeypatch.setattr(api.query_limiter, "_clients", {})
