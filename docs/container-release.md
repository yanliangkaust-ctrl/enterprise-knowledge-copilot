# Phase 4B container release gate

No image is published and no cloud deployment is performed by this workflow.

## Dependency derivation

requirements-release.txt pins the installed Phase 3B versions of FastAPI, NumPy,
NetworkX, OpenAI, LangGraph, Uvicorn and HTTPX and their transitive dependencies.
The dependency closure was derived from installed distribution Requires-Dist
metadata evaluated for Linux/Python 3.12. All selected distributions' Requires-Python
metadata permits 3.12. requirements-release-dev.txt adds the installed pytest set.
No upgrades were made. Existing broad requirements remain for the optional local UI.
The plain Uvicorn distribution supports the existing HTTP service without optional
standard extras. Streamlit, pyvis, sentence-transformers and torch are not needed in
the hash-only release image. LangSmith is an existing LangGraph dependency, not an
enabled external evaluation service; do not set tracing/cloud credentials.

Exact versions are reproducible pins, not a hash-verified supply-chain lock. Python
3.12 wheel availability, installation and runtime are verified by CI, not inferred
from successful Windows/Python 3.14 tests. The python:3.12-slim base tag is mutable;
retain the built image ID/digest for the release rather than assuming rebuild identity.

## Bundled corpus

Only data/sample_docs is copied:

- ADR_Deployment.md
- API_Specification.md
- Deployment_Guide.md
- Incident_Batch_Failure.md
- OCR_Platform_Architecture.md
- Risk_Register.md
- Security_Requirements.md
- UAT_Report.md

These are the existing synthetic/demo OCR corpus, reviewed for this release. No
private/personally identifying content or credentials were found in these files.
No evaluation reports, Cobalt Relay document, developer database, or other data files
ship. Do not add private files to this directory. Startup seeds missing filenames only.

## Validation

Run on a Docker host after the Python test/evaluation gates:

```sh
docker build --tag enterprise-knowledge-copilot-api:COMMIT_SHA .
python scripts/validate_container.py --image enterprise-knowledge-copilot-api:COMMIT_SHA
```

The validator creates uniquely named disposable Docker volumes/containers, binds
8000 to a random localhost host port, waits at most 90 seconds for readiness and
checks supported/unsupported queries and unique UUID request headers. It adds and
updates a synthetic Quartz Relay record through the existing KnowledgeBase ingestion
method via docker exec. It verifies revision, UUID, v2 and query behavior after both
same-container restart and new-container replacement using the exact same image.
Rebuilding a second image is not required: retaining exact image identity is the gate.

Storage outage is tested by temporarily replacing the disposable database path with
a directory, then restoring it and checking 503/readiness/recovery. A read-only-root
container without a disk must fail startup. Stops must exit zero and log completed
application shutdown. Only structured application events are checked as JSON;
Uvicorn's own lifecycle/access logs remain plain text. Logs and the machine-readable
report are saved under artifacts/container-validation and resources are cleaned up.
Docker subprocesses and HTTP/startup waits are bounded.

CI installs the pinned dependencies, runs the full suite and exact evaluation gate,
builds an image tagged with Git SHA plus OCI revision label, then validates that image.
Failed tests/evaluation/build/smoke checks fail the workflow. Evaluation and container
artifacts are retained. Image tags are only locally created; future publication must
use commit tags and deploy a recorded immutable registry digest.

Defaults: /knowledge, hash embeddings, port 8000, one default Uvicorn worker. No
OPENAI_API_KEY is passed; the image contains no keys. The API uses local generation.
No model download is required. /knowledge must be mounted writable for persistence;
no VOLUME declaration creates hidden anonymous state.

Until the Docker-host workflow passes, build/startup/health/readiness/query/IDs/
persistence/shutdown remain UNVERIFIED, regardless of local Python test results.
Before cloud deployment commit all release files, inspect successful CI artifacts,
choose the single-instance disk-backed platform, configure HTTPS and /ready checks,
and validate a consistent SQLite backup/restore and rollback procedure.
