# Phase 2A local evaluation

Run from the repository root:

```console
python -m backend.evaluation_runner --check
python -m pytest -q tests/test_evaluation_harness.py tests/test_retrieval.py tests/test_sprint5.py
python -m pytest -q
```

The gate writes `phase2_report.json`. CI uploads it as `rag-evaluation-report`.
The original ten-question fixture is unchanged. The supplemental fixture adds four
answerable questions and four unsupported questions, plus two fixed documents.
Both suites use the same fresh in-memory corpus, hash vectors, and local generation.
No SQLite store, user upload, API key, model download, or existing session is used.

## Metric contract

Each strategy executes once per question with K=5. Metrics slice that frozen ranking
at 1, 3, and 5. Positions are result slots, not reranked distinct-document positions.

- Hit Rate@K: 1 if any expected source occurs, otherwise 0.
- Document Recall@K: distinct expected sources found / expected source count.
- Document Precision@K: distinct expected sources found / K.
- MRR@5: reciprocal rank of first expected-source result within five slots, or 0.
- Evidence Recall/Precision: same recall/precision arithmetic over unique labeled
  chunks, available only for supplemental cases.

Duplicates and missing slots receive no extra precision credit. Consequently a
single-source question has maximum document Precision@5 of 0.2. Positive aggregate
metrics are macro averages over answerable questions; unsupported cases have zero
retrieval scores and are excluded from those averages. Outcome rates state their
denominators. Required-fact coverage is micro averaged over labeled facts and uses
normalized literal alternatives, not semantic entailment. Refusals get no fact credit.
Source-supported fact coverage additionally checks that labeled evidence was retrieved
and its document is in the response's source list; it is not claim-level citation validation.

Gold selectors use filename, section, and SHA-256 of whitespace-normalized text.
Selectors must resolve exactly once. Reports retain evidence text hashes instead of
duplicating full chunk text. `non-gold` means outside the fixture's specified support;
the original suite has document-only judgments and no chunk oracle.

`recall_at_k` and integer-key `evaluate_retrieval` now calculate true recall.
The former any-hit behavior is available as `hit_rate_at_k`. `mean_reciprocal_rank`
now uses five slots. The legacy question-type `MRR` key aliases `MRR@5` for callers.
The Evaluation UI uses chunk keyword retrieval to match the runtime strategy.

## Baseline and gate

`phase2_baseline.json` is the initial regression reference captured for this change.
It records current failures without treating them as acceptable overall RAG quality.
The designated critical contract preserves Cobalt Relay vector Hit@1, its accepted
local answer and required protocol fact, plus refusal of the unrelated budget question.
Only decreases on these fields are quality regressions that fail this initial gate.

The gate also fails invalid/empty benchmarks, unresolved gold, evidence metadata
integrity errors, source references outside retrieved evidence, and unequal repeated
local runs. Corpus/benchmark/configuration identity changes require explicit review
and rebaselining; Python/package versions are recorded but not required to match.
CI separately runs the full existing test suite. No universal Recall/Precision/MRR
floor or unsupported-acceptance floor is imposed in Phase 2A.

To intentionally update a baseline after review, run the harness without `--check`,
then copy its identity, configuration, aggregates, outcomes, and critical fields into
the baseline. CI never updates the baseline automatically.

Known initial quality failures: two related-entity unsupported questions are accepted;
the endpoints and multi-source supplemental answers miss their labeled facts.
Original questions q08 and q10 are refused. Retrieval and sufficiency are unchanged.
The small fixture and hash backend do not certify transformer or OpenAI generation quality.
