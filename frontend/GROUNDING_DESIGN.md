# Current implementation update — final sprint

The Phase 5E inventory below is historical. The final sprint now implements a synthetic-only bounded corpus graph API, literal claim/value checks with structured results, and high-risk signaling. General entailment, calibrated confidence, authenticated human approval and durable review queues remain absent. SUPPORTED means literal source agreement only. Conditional excerpts retain NOT_VERIFIED; matching subject/value text is not a full verification of the omitted condition. Simulated operational text is not verified against document citations. See `backend/claim_verification.py`, `backend/public_graph.py` and `/query` additive fields. Docker checks are required before release.

# Grounding and verification inventory — Phase 5E

This document describes inspected code, not production certification. No backend behavior or API fields are added in this phase.

## Existing safeguards

| Capability | State | Code and limits |
| --- | --- | --- |
| Retrieval and support layering | Implemented, bounded | `backend/langgraph_orchestrator.py`: routing, graph weakness checks, fallback, draft generation, sufficiency, refusal, finalization. No universal semantic guarantee. |
| Requested factual facets | Implemented, partial coverage | `backend/factual_support.py` and `backend/agentic_orchestrator.py`: deterministic question/evidence support checks. They do not audit each generated claim. |
| Procedural support | Implemented, conservative | `backend/procedural_support.py` and `backend/answer_generation.py`: shared action/structure support predicate, refusal, extractive supported procedure output. Lexical scope is limited. |
| Ownership/relationship support | Implemented, bounded | Graph support or explicit textual ownership when graph resolution is unavailable; topic overlap alone does not suffice. |
| Post-draft sufficiency gate | Implemented | `evaluate_evidence` creates a draft then calls `_evidence_sufficient`; it verifies question/evidence support, not individual claims or numeric/date agreement in answer text. Operational text can be appended afterward. |
| Source provenance | Implemented, partial attribution | API returns `sources`, chunk IDs/text/rank/section/score and graph metadata where available. No sentence/claim-to-citation alignment. |
| OpenAI generation path | Partial protection | Grounded prompt and final sufficiency gate; no independent claim verification. `/query` currently uses local generation. |
| Offline deterministic evaluation | Implemented | Arithmetic, evidence integrity, outcome labels and designated gates in `backend/evaluation_runner.py`; offline tests are not runtime claim verification. |
| Numeric/date consistency | Not implemented | Facet support is not equality/unit/date comparison between generated answer and sources. |
| Claim-level post-generation verification | Not implemented | No claim objects, verifier results or per-claim support/contradiction status in `/query`. |
| Human approval | Not implemented | No reviewer identities, authenticated review workflow, queue, approval persistence or enforcement. Corpus text mentioning approval is not an application approval state. |
| General entailment or confidence probability | Not implemented | Neither retrieval scores nor grounding labels are calibrated truth probabilities. |

## Truthful inspector

The Sources tab uses only `sources` and returned `provenance`, and labels uncited refusal evidence explicitly. It displays source chunks, not full documents or invented download links. Verification shows literal API sufficiency, grounding status, route, reason and supplied execution trace. There are no per-request green checks for factual facets/procedure checks because the API does not expose each check's result. Unimplemented verification and approval always show unavailable/not evaluated.

Knowledge Graph displays only evidence with nonempty `relationship_type` and exactly two explicit `entity_path` names. `backend/graph_retrieval.py` defines that pair in source-to-target order. No edges are inferred from text, filenames, selection reasons, shared entities, or hybrid scores. These are answer-specific retrieved corpus relationships, not claim verification. A refusal's graph is retrieval context. Hybrid evidence currently lacks typed edge metadata; an empty graph view says metadata is unavailable rather than asserting no relationships exist. Full corpus overview requires a new endpoint and is unavailable now.

The checked-out starting UI had two columns. Phase 5E adds a third inspector column using the same forest-green/off-white design. Latest turns render first; the inspector always follows the latest turn of the active local conversation. Pending/error turns clear previous inspector results. At narrower widths the inspector stacks beneath chat. Client conversations remain in-memory and API/error handling is preserved.

## Minimal future backend design (proposal only)

1. **Versioned claim verification.** After complete answer assembly (including operational additions), create claims with exact answer character spans. Start with deterministic extractive statements and explicit supported forms, not general semantic entailment. Each claim references existing chunk IDs and exact supporting source spans; statuses are `supported`, `contradicted`, `unverified` with rule/version/reason. Unsupported parsing stays unverified. Freeze evidence against a knowledge revision and answer hash. Verifier exceptions fail closed; tests cover unsupported additions, wrong subject, mixed supported/unsupported sentences, citation mismatches and final operational text. Initially shadow-report results to avoid silently changing established Phase 2 behavior; enforcing refusal is a separately reviewed change.
2. **Bounded numeric/date checks.** Only compare identified claim values to values for the same subject and facet in cited spans. Normalize supported units and unambiguous dates explicitly, preserving original forms and units. Ambiguous dates, ranges, computed totals, conversions and multiple competing values remain unverified until rules exist. Tests cover incorrect amount/date/unit, unrelated matching number, ranges, ambiguity and evidence revision changes. Do not label the full answer verified merely because one number matches.
3. **Optional human review.** A versioned high-risk policy flags decisions as `review_required`; it must not imply approval. A real workflow needs authentication/authorization, reviewer identity, timestamps, answer/evidence hashes, knowledge revision, decision reason and an auditable queue (a transactional SQLite review table may suffice for a single-instance persistent deployment). Approvals are bound to the exact answer version and invalidated on changes. No approval button without this backend enforcement. Render Free ephemeral storage cannot provide durable review records. Tests cover unauthorized reviewers, duplicate decisions, pending/rejected states, stale approvals and restart durability.

Proposed additive `/query` fields: `knowledge_revision`, `answer_hash`, `verification` containing verifier version, evaluation status, claim spans/support references/reasons and numeric/date checks; optional `review` with `not_required|required|pending|approved|rejected` and review ID. Add queue creation/list/decision endpoints only when authenticated review is implemented. The frontend must treat missing new fields as unavailable, never verified or approved. A corpus overview would separately need a revisioned graph endpoint exposing edge provenance. No such API, queue or table exists today.
