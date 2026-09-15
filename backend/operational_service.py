"""Deterministic simulated operational data for incident status queries."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class IncidentStatus:
    incident_id: str
    service: str
    status: str
    severity: str
    owner: str
    last_updated: str
    current_action: str
    simulated: bool = True


_INCIDENTS = {
    "INC-001": IncidentStatus(
        incident_id="INC-001",
        service="OCR Service",
        status="Mitigated",
        severity="SEV-2",
        owner="Platform Engineering Team",
        last_updated="2025-04-18T16:30:00Z",
        current_action="Monitoring replayed batch jobs and validating queue depth.",
    ),
}


def extract_incident_id(text: str) -> str | None:
    match = re.search(r"\bINC-\d{3}\b", text.upper())
    return match.group(0) if match else None


def get_incident_status(incident_id: str) -> IncidentStatus | None:
    """Return simulated operational status; unknown incidents return no result."""

    return _INCIDENTS.get(incident_id.upper())


def format_incident_status(status: IncidentStatus) -> str:
    return (
        f"Simulated operational data for {status.incident_id}: {status.service} is "
        f"{status.status} ({status.severity}). Owner: {status.owner}. "
        f"Last updated: {status.last_updated}. Current action: {status.current_action}"
    )