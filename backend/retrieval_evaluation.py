"""Deterministic retrieval metrics over ranked result slots.

Document precision uses unique relevant documents / K. Missing slots and repeated
documents receive no credit. Recall uses unique relevant documents / gold count.
"""
from collections.abc import Callable, Mapping, Sequence
from backend.document_ingestion import DocumentRecord

RetrievalFunction = Callable[[str, int], Sequence[object]]


def validate_questions(questions: Sequence[dict]) -> None:
    if not questions:
        raise ValueError("Benchmark must be non-empty")
    ids = set()
    for question in questions:
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError("Question must contain non-empty text")
        sources = question.get("expected_sources")
        if not isinstance(sources, list) or any(not isinstance(s, str) or not s for s in sources):
            raise ValueError("expected_sources must be a list of document names")
        if len(sources) != len(set(sources)):
            raise ValueError("Duplicate expected sources")
        if not sources and question.get("answerable") is not False:
            raise ValueError("Empty gold sources require answerable=false")
        if "id" in question:
            if not isinstance(question["id"], str) or not question["id"] or question["id"] in ids:
                raise ValueError("Question IDs must be unique non-empty strings")
            ids.add(question["id"])


def _check_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("K must be a positive integer")


def metrics_for_results(expected_sources: Sequence[str], results: Sequence[object], ks=(1, 3, 5)) -> dict[str, float]:
    """Score one frozen ranking; no-gold cases score zero, excluded from positive aggregates."""
    gold = set(expected_sources)
    metrics = {}
    for k in ks:
        _check_k(k)
        names = {_result_document_name(result) for result in results[:k]}
        found = len(names & gold)
        metrics[f"Hit Rate@{k}"] = float(found > 0)
        metrics[f"Recall@{k}"] = found / len(gold) if gold else 0.0
        metrics[f"Precision@{k}"] = found / k
    metrics["MRR@5"] = next((1 / rank for rank, result in enumerate(results[:5], 1)
                              if _result_document_name(result) in gold), 0.0)
    return metrics


def evaluate_metrics(questions: Sequence[dict], retrieve: RetrievalFunction, ks=(1, 3, 5)) -> dict[str, float]:
    """Fetch once per query, then macro-average positive query metrics."""
    validate_questions(questions)
    ks = tuple(ks)
    for k in ks:
        _check_k(k)
    rows = [metrics_for_results(q["expected_sources"], list(retrieve(q["question"], max((5, *ks)))), ks)
            for q in questions if q["expected_sources"]]
    if not rows:
        raise ValueError("Retrieval aggregate requires at least one positive question")
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}


def recall_at_k(questions, retrieve, k):
    return evaluate_metrics(questions, retrieve, (k,))[f"Recall@{k}"]


def hit_rate_at_k(questions, retrieve, k):
    return evaluate_metrics(questions, retrieve, (k,))[f"Hit Rate@{k}"]


def precision_at_k(questions, retrieve, k):
    return evaluate_metrics(questions, retrieve, (k,))[f"Precision@{k}"]


def evaluate_retrieval(questions, retrieve, ks=(1, 3, 5)):
    """Preserve integer-key API; values now correctly represent document recall."""
    ks = tuple(ks)
    metrics = evaluate_metrics(questions, retrieve, ks)
    return {k: metrics[f"Recall@{k}"] for k in ks}


def mean_reciprocal_rank(questions, retrieve):
    """MRR at an explicit cutoff of five result slots."""
    return evaluate_metrics(questions, retrieve)["MRR@5"]


def evaluate_by_question_type(questions, retrieve, ks=(1, 3, 5)):
    validate_questions(questions)
    ks = tuple(ks)
    grouped = {}
    for question in questions:
        grouped.setdefault(question["question_type"], []).append(question)
    metrics = {kind: evaluate_metrics(group, retrieve, ks) for kind, group in grouped.items()}
    # Compatibility key used by older callers; the UI displays the explicit MRR@5.
    return {kind: {**values, "MRR": values["MRR@5"]} for kind, values in metrics.items()}


def _result_document_name(result: object) -> str:
    if isinstance(result, Mapping):
        return str(result.get("document_name", result.get("source_document", result.get("name", ""))))
    if isinstance(result, DocumentRecord):
        return result.name
    chunk = getattr(result, "chunk", None)
    if chunk is not None:
        return chunk.document_name
    return getattr(result, "document_name", getattr(result, "source_document", ""))
