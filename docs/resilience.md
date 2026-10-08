# Local resilience contract

Successful retrieval and evaluation behavior is unchanged. No retries or stale
serving are implemented.

In-memory rebuilds prepare documents, chunks, vectors and graph before publication.
The existing VectorIndex object retains its identity; its chunk/vector pair is
swapped atomically and searches capture that pair before encoding the query.
Persistent updates continue to prepare inside the SQLite transaction and publish
only after commit. Failed reloads retain the old snapshot but refuse to serve it.

ServiceFailure carries a fixed code and HTTP status, never original exception text.
Storage/revision/reload failures return generic 503 bodies; pipeline failures return
generic 500 bodies. API request IDs and sanitized structured events remain available.
Both FastAPI and Streamlit initialization sanitize exceptions and remain fail-safe.

Hybrid independently calls each component once. One failed component can use the
healthy component's results with existing weights and coverage selection. Two failed
components, or a failed component with no remaining evidence, yield service failure.
A graph/vector primary failure can call the independent alternative once, without
weak-result recursion. All degraded evidence passes unchanged final support checks.
Degradation emits retrieval_degraded and records the failed component/final route;
it does not fabricate graph provenance or renormalize scores.

Readiness remains read-only and bounded. Cached reload/revision failures and known
unusable runtime retrieval/generation states return 503. A successful real query
clears the runtime marker; a successful snapshot/update clears storage failure state.
Readiness never probes generation or retries retrieval. Markers are process-local:
concurrent request completions can supersede earlier observations.

The final API response is built and serialization checked before session recording.
Transport delivery after recording is not guaranteed: there is no delivery transaction,
streaming recovery, or session redesign. Concurrent in-memory knowledge requests should
take a fresh snapshot; retained index references intentionally keep existing identity.

Remaining limitations include existing SQLite waits/request lock contention, no
backend health polling, no retries/circuit breakers, no multi-worker session store,
and no stale serving. Arbitrary component exceptions are classified at the retriever
boundary; degradation is constrained to independent evidence and the existing gates,
not a general semantic correctness guarantee.
