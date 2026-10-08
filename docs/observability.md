# Local application observability

FastAPI and Streamlit enable the `enterprise.observability` logger at their entry
points. It writes JSON lines to stdout using only standard-library logging.
Library imports remain quiet, preserving subprocess and evaluation CLI output.
Call `configure_json_logging()` explicitly to enable events in another entry point.

Every HTTP request receives an authoritative UUID in `X-Request-ID`, including
validation and server failures. Client-supplied request IDs are ignored. Context
propagates to synchronous endpoint and orchestration work; session IDs are not used
as correlation IDs. Startup events and Streamlit work have a null request ID.

## Events and timings

Events include `request_started`, `request_completed`, `stage_completed`,
`retrieval_completed`, `retrieval_candidates`, `hybrid_candidates`, `hybrid_selection`,
`graph_support_check`, `graph_fallback`, `factual_support_check`, `sufficiency_check`,
`relationship_support`, `generation_refusal`, `knowledge_initialized`,
`knowledge_snapshot`, `knowledge_reload`, `readiness_check`, and `application_error`.

UTC timestamps mark events; non-negative milliseconds use `perf_counter` durations.
Stages cover initialization, snapshot, reload, context, orchestration, routing,
primary/fallback retrieval, vector/graph/keyword retrieval, hybrid selection,
factual support, sufficiency, draft generation, response/refusal finalization, and
trace finalization. Nested durations overlap: do not sum them into total latency.
The request completion duration includes response sending. `draft_generation`
wraps the actual generator call inside `evaluate_evidence`, not the later node.

Retrieval events expose requested/effective K, initial/final route, candidate counts,
merged unique counts, selected/reserved counts, and bounded min/max score summaries.
Vector candidates mean indexed chunks scored; keyword candidates mean positive-score
chunks; graph candidates mean relationship rows before truncation. Hybrid events
separately count component results and merged unique chunks.

Graph checks expose `graph_empty`, `graph_weak_text_support`,
`graph_missing_fact_support`, and `graph_support_adequate`. Final sufficiency codes
include `no_evidence`, `missing_question_support`, `missing_factual_support`,
`missing_relationship_support`, `missing_risk_support`, `missing_recipient_support`,
`missing_team_support`, `missing_release_check_support`, `evidence_sufficient`,
`operational_record_found`, and `operational_record_not_found`.
Ownership policy may log `explicit_textual_ownership` after the strict facet check;
this records the existing policy alternative without changing the predicate.
Graph-check codes also appear alongside existing API retrieval-check explanations.

## Knowledge state and probes

Snapshot events capture the loaded revision while holding the snapshot lock, so
the request summary reports the revision actually used, not a later global value.
Reload events contain prior/new revision, duration, document/chunk counts, and
embedding backend; failures contain a type and fixed code without exception text.
In-memory evaluation knowledge uses revision -1.

`GET /health` remains the original static liveness response.
`GET /ready` returns 200 only when an initialized snapshot/index is available and
the persistent store revision is accessible and matches the loaded revision.
Otherwise it returns 503 with `knowledge_uninitialized`, `knowledge_busy`,
`index_unavailable`, `store_unavailable`, or `knowledge_stale`.
Readiness obtains the lock for at most 50 ms and uses a read-only SQLite connection
with a 100 ms busy timeout for one revision-row query. It never initializes knowledge,
reloads indexes, downloads a model, or runs a benchmark. A stale process becomes ready
after its normal request-driven snapshot reload. These timeouts bound lock waits,
not arbitrary operating-system filesystem stalls.

## Content and error policy

The event field allowlist excludes questions, answers, document text, full traces,
provenance, credentials, session IDs, validation bodies, and arbitrary exception
messages. Subject anchors are not logged. Only bounded primitive metadata is emitted.
Application errors before response headers receive the existing generic HTTP 500
body and a request ID; sensitive exceptions are not rethrown into Uvicorn's traceback
logger. Failures after response headers use a sanitized exception.

Existing API evidence/trace and validation responses are unchanged and can contain
content. Do not treat them as safe log payloads. Uvicorn access logs and third-party
library logs retain their own configuration; this module is not a global redactor.
No log retention, collector, metrics service, alerting, or distributed tracing is
provided. Readiness is a lightweight availability check, not a full integrity audit.

Regression artifacts live in `data/evaluation/phase3a_before.json`,
`phase3a_after.json`, and `phase3a_comparison.json`; the Phase 2 reports are identical.
