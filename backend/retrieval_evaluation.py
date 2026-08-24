"""Recall evaluation for keyword and semantic retrieval modes."""

from collections.abc import Callable, Sequence
from collections.abc import Mapping

from backend.document_ingestion import DocumentRecord


RetrievalFunction = Callable[[str, int], Sequence[object]]


def recall_at_k(
    questions: Sequence[dict],
    retrieve: RetrievalFunction,
    k: int,
) -> float:
    """Calculate source-document recall for a set of questions at rank k."""

    if not questions:
        return 0.0
    hits = 0
    for question in questions:
        results = retrieve(question["question"], k)
        names = {
            _result_document_name(result)
            for result in results
        }
        expected_sources = set(question["expected_sources"])
        if names.intersection(expected_sources):
            hits += 1
    return hits / len(questions)


def evaluate_retrieval(
    questions: Sequence[dict],
    retrieve: RetrievalFunction,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[int, float]:
    """Return Recall@K values as proportions for the requested cutoffs."""

    return {k: recall_at_k(questions, retrieve, k) for k in ks}


def mean_reciprocal_rank(questions: Sequence[dict], retrieve: RetrievalFunction) -> float:
    """Calculate MRR using the first retrieved result from an expected source."""

    if not questions:
        return 0.0
    reciprocal_ranks = []
    for question in questions:
        expected_sources = set(question["expected_sources"])
        rank = 0
        for index, result in enumerate(retrieve(question["question"], 100), start=1):
            if _result_document_name(result) in expected_sources:
                rank = index
                break
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_by_question_type(
    questions: Sequence[dict],
    retrieve: RetrievalFunction,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, dict[str, float]]:
    """Return Recall@K and MRR for each labeled question type."""

    grouped: dict[str, list[dict]] = {}
    for question in questions:
        grouped.setdefault(question["question_type"], []).append(question)
    return {
        question_type: {
            **{f"Recall@{k}": recall_at_k(group, retrieve, k) for k in ks},
            "MRR": mean_reciprocal_rank(group, retrieve),
        }
        for question_type, group in grouped.items()
    }


def _result_document_name(result: object) -> str:
    if isinstance(result, Mapping):
        return str(result.get("document_name", result.get("name", "")))
    if isinstance(result, DocumentRecord):
        return result.name
    chunk = getattr(result, "chunk", None)
    if chunk is not None:
        return chunk.document_name
    return getattr(result, "document_name", "")
