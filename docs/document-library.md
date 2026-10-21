# Document Upload & Ingestion MVP

The React Document Library and FastAPI endpoints share the existing SQLite
KnowledgeBase, section chunker, vector index, retrieval graph and query pipeline.
No retrieval mode, evidence gate, claim verifier, or sample corpus is replaced.

## API

- `GET /documents`: active filename, UUID, version, ready status, text byte size,
  chunk count, and a shared/ephemeral storage notice.
- `POST /documents?filename=example.pdf`: raw file body with Content-Type
  `application/octet-stream` (PDF/plain-text/Markdown MIME types also accepted).
  Returns 201 only after transactional indexing, with the same document metadata.
  Exact filename/extracted-text retries are no-ops; changed content/name casing
  collisions return 409. Renaming is required for different content.
- Query through unchanged `POST /query`. No restart is required.

Limits: 1 MiB incoming file bytes; 200,000 extracted UTF-8 bytes; 20 PDF pages;
28 active documents including the eight samples; 2,000,000 corpus UTF-8 bytes.
One upload at a time per process; six upload attempts/client/minute. Upload-body
receipt has a 30-second timeout; PDF extraction has a 15-second subprocess timeout.
On Linux the PDF worker also has a 256 MiB address-space limit. These are demo
bounds, not a general hostile-file processing service. Use one Uvicorn worker.
Storage limits/collisions are checked inside the SQLite write transaction.

PDF extraction uses pinned pypdf 6.21.0. Only embedded text is extracted; no OCR,
layout/table reconstruction, or password handling. Empty, corrupt, encrypted,
scanned/image-only PDFs fail with actionable errors. DOCX is unsupported.
TXT and Markdown must be UTF-8; BOM is accepted. Originals are not retained:
the SQLite content column contains extracted text under the original filename.
For PDF, citations identify filename/chunk, not page coordinates.

The shared text chunker now handles heading-free text and Markdown preambles.
Existing headed sample chunks retain their original content/order and IDs.
All eight sample filenames are reserved by the upload API. Existing local admin
ingest_markdown behavior is retained for compatibility. The public `/graph`
overview remains synthetic-only; uploaded relationships may appear in query evidence.

## Deployment and privacy

Install the updated requirements; the Dockerfile already installs
requirements-release.txt. No new external service, environment variable, CORS
method or header is needed: the current exact-origin GET/POST policy supports
the upload API. Deploy backend and frontend together only after review.

This MVP has the application's existing shared, unauthenticated demo model.
Every visitor can query uploaded content and see filenames. The UI explicitly
requires synthetic documents; it is not a private enterprise library. Before
accepting confidential documents add authentication, authorization and isolation.

Repository Render configuration uses `/knowledge` on Free compute without a
persistent disk. SQLite writes survive while that filesystem survives, but Free
restart/redeploy/idle spin-down can discard uploads. Startup restores only the
eight bundled samples. No durable persistence is claimed. Account-side storage
settings were not changed or independently inspected during implementation.

Smallest durable option: paid single-instance Render service with a persistent
disk mounted at `/knowledge`, retaining KNOWLEDGE_STORE_DIR=/knowledge, plus
SQLite-consistent backup/restore validation. Free compute with external durable
storage requires a separate storage integration; browser storage is not a backup.

References: https://render.com/docs/free and https://render.com/docs/disks.

## Verification

`tests/test_uploads.py` sends actual bytes through TestClient for MD, TXT and PDF,
queries a unique Violet Beacon/Zircon Thread fact, checks filename/versioned
evidence references, then verifies an unsupported price question refuses. Tests
also cover parser errors, empty/invalid/oversized files, protected sample names,
duplicate handling and indexing rollback. Existing regression tests run unchanged.

The bounded verifier may return NOT_VERIFIED for generated headings or other
uncovered text; this feature does not upgrade those results to SUPPORTED.
Deletion, replacement UI, bulk upload, DOCX, private workspaces and OCR are deferred.
