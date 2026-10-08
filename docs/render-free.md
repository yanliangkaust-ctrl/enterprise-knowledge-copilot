# Render Free public demo

Deploy only the FastAPI Docker service. Free storage is disposable: SQLite and
session history are lost on restart/redeploy/idle spin-down. The eight bundled
synthetic documents are recreated automatically. This is a restored sample corpus,
not durable user knowledge. UUIDs/revisions may change after storage resets.
Cobalt Relay and developer databases do not ship. No public upload API exists.

Render Free currently sleeps after 15 idle minutes; waking typically takes about
one minute. Its 512 MB RAM / 0.1 CPU allocation requires actual startup and memory
validation. See https://render.com/docs/free and
https://render.com/docs/compute-plans for current platform limits.

## Exact settings

- Web Service; repository enterprise-knowledge-copilot; verified green main commit.
- Runtime Docker; root directory blank; Dockerfile ./Dockerfile; build context root.
- Free compute; one instance; Docker command override blank; no persistent disk.
- Health-check path /ready; retain /health as liveness.
- KNOWLEDGE_STORE_DIR=/knowledge; KNOWLEDGE_EMBEDDING_BACKEND=hash; PORT=8000.
- Leave OPENAI_API_KEY and cloud tracing credentials unset.
- Keep auto-deploy disabled until the release CI has passed; deploy manually.
- Platform HTTPS; do not enable wildcard CORS. No frontend CORS is needed yet.

## Resource bounds

POST /query accepts 1–2000 characters; session identifiers are limited to 128
characters. Invalid/oversized fields return 422 before orchestration.
The process stores at most 256 sessions, five turns each; inactive sessions expire
after 1800 seconds. Expiration is lazy on access/record; capacity evicts the least
recently accessed session. Expired/evicted conversation IDs start with empty context.

POST /query is limited to 30 requests per ASGI client address per fixed 60-second
window, including malformed requests. 429 contains a fixed reason code, Retry-After,
and X-Request-ID. Other endpoints are exempt. The limiter stores at most 1024 client
windows; at capacity new clients receive 429 until a window expires. Locking protects
concurrent counters. No queries, IP addresses or session identifiers are logged by
the limiter.

Client identification uses scope.client, not application parsing of forwarded
headers. Uvicorn/platform trusted proxy configuration determines that address.
Verify distinct real clients on Render: a shared proxy address can group visitors,
and trusting arbitrary forwarded headers permits spoofing. Do not set unrestricted
proxy trust. Limits are process-local, reset on restart, and are not authentication.
Shared NAT users share a budget; fixed windows allow bursts across a boundary.

No total HTTP body-size cap, global concurrency limit, authentication, or distributed
abuse protection is introduced here. Rate limits/session bounds reduce exposure but
do not guarantee memory safety for arbitrary large bodies or generated evidence.

## Release checklist

1. Full Python tests and exact deterministic evaluation must pass unchanged.
2. Existing CI must build and run scripts/validate_container.py against the image.
3. Require free_eight_document_reconstruction and existing persistent-volume checks.
4. Deploy the verified commit manually; check HTTPS /health, /ready and /query.
5. Verify supported answer, unsupported refusal, oversized 422, and bounded 429.
6. Check request IDs, Retry-After, logs, client-address behavior and memory usage.
7. Restart/redeploy: verify the eight sample documents reconstruct; expect lost history.
8. Inform users of cold starts and ephemeral demo data. Never add private corpus.

Paid mode remains unchanged: attach a writable persistent disk at /knowledge and use
the same settings. Keep persistent rollback/version/restart tests. Do not migrate
SQLite, disable its transactions, or remove the paid persistence path.
