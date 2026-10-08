from copy import deepcopy

import pytest

from backend.evaluation_runner import (check_baseline, integrity_errors, load_fixture,
                                       relevance_metrics, resolve_selector, run_evaluation)
from backend.rag_schema import EvidenceItem
from backend.retrieval_evaluation import (evaluate_metrics, hit_rate_at_k, metrics_for_results,
                                         mean_reciprocal_rank, precision_at_k, recall_at_k,
                                         validate_questions)


def results(*names):
    return [{"document_name": name} for name in names]


def test_multiple_sources_duplicates_and_irrelevant_tail_exact_arithmetic():
    metrics = metrics_for_results(["a", "b"], results("x", "a", "a", "b", "z", "c"))
    assert metrics == {
        "Hit Rate@1": 0, "Recall@1": 0, "Precision@1": 0,
        "Hit Rate@3": 1, "Recall@3": 0.5, "Precision@3": 1 / 3,
        "Hit Rate@5": 1, "Recall@5": 1, "Precision@5": 2 / 5, "MRR@5": 0.5}


def test_short_rankings_do_not_change_precision_denominator():
    m = metrics_for_results(["a", "b"], results("a"))
    assert m["Recall@5"] == 0.5
    assert m["Precision@5"] == 0.2
    assert m["MRR@5"] == 1
    assert all(value == 0 for value in metrics_for_results(["a"], []).values())
    assert metrics_for_results(["a"], results("x", "x", "x", "x", "a"))["MRR@5"] == 0.2
    assert metrics_for_results([], results("a"))["MRR@5"] == 0


def test_explicit_cutoff_even_if_retriever_ignores_k():
    q = [{"question": "test", "expected_sources": ["a"]}]
    retrieve = lambda question, k: results("x", "x", "x", "x", "x", "a")
    assert hit_rate_at_k(q, retrieve, 5) == 0
    assert recall_at_k(q, retrieve, 5) == 0
    assert precision_at_k(q, retrieve, 5) == 0
    assert mean_reciprocal_rank(q, retrieve) == 0


def test_macro_averages_and_single_fetch_per_query():
    questions = [{"question": "one", "expected_sources": ["a", "b"]},
                 {"question": "two", "expected_sources": ["c"]}]
    calls = []
    def retrieve(question, k):
        calls.append((question, k))
        return results("a") if question == "one" else results("x", "c")
    m = evaluate_metrics(questions, retrieve)
    assert calls == [("one", 5), ("two", 5)]
    assert m["Recall@1"] == 0.25
    assert m["Recall@3"] == 0.75
    assert m["Hit Rate@1"] == 0.5
    assert m["Precision@3"] == pytest.approx(1 / 3)
    assert m["MRR@5"] == 0.75


@pytest.mark.parametrize("questions", [[], [{"question": "", "expected_sources": ["a"]}],
    [{"question": "q", "expected_sources": []}],
    [{"question": "q", "expected_sources": ["a", "a"]}],
    [{"question": "q", "expected_sources": "a"}],
    [{"id": "same", "question": "q", "expected_sources": ["a"]}] * 2])
def test_invalid_benchmark_rejected(questions):
    with pytest.raises(ValueError):
        validate_questions(questions)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoff_rejected(k):
    with pytest.raises(ValueError):
        metrics_for_results(["a"], results("a"), (k,))


def test_negatives_excluded_from_positive_aggregate():
    qs = [{"question": "yes", "expected_sources": ["a"]},
          {"question": "no", "expected_sources": [], "answerable": False}]
    assert evaluate_metrics(qs, lambda q, k: results("a"))["Recall@1"] == 1


def test_integrity_rejects_fabricated_metadata_and_sources():
    _, _, _, chunks = load_fixture()
    c = chunks[0]
    e = EvidenceItem(rank=1, source_document=c.document_name, section=c.section_heading,
                     chunk_id=c.chunk_id, text=c.text, score=1)
    assert integrity_errors([e], [c.document_name], chunks) == []
    for field in ("source_document", "section", "chunk_id", "text"):
        assert integrity_errors([e.model_copy(update={field: "fabricated"})], [], chunks)
    assert integrity_errors([e], ["fabricated.md"], chunks) == ["Fabricated source reference"]


def test_selector_resolves_without_runtime_number_and_rejects_stale_text():
    _, supplemental, _, chunks = load_fixture()
    selector = supplemental["questions"][0]["gold_evidence"][0]
    original = resolve_selector(selector, chunks)
    from dataclasses import replace
    renamed = [replace(c, chunk_id="new-id") if c.chunk_id == original else c for c in chunks]
    assert resolve_selector(selector, renamed) == "new-id"
    with pytest.raises(ValueError, match="Unresolved"):
        resolve_selector({**selector, "text_sha256": "stale"}, chunks)


def test_evidence_precision_detects_correct_hit_with_distractors():
    evidence = [type("E", (), {"chunk_id": name})() for name in ("gold", "x", "y", "z", "w")]
    assert relevance_metrics(evidence, {"gold"})["Evidence Precision@5"] == 0.2
    assert relevance_metrics(evidence, {"gold"})["Evidence Recall@5"] == 1


def test_harness_repeatability_and_current_failures_are_visible():
    first = run_evaluation()
    assert first == run_evaluation()
    assert first["integrity_errors"] == []
    assert len([q for q in first["queries"] if q["suite"] == "original"]) == 10
    assert first["outcomes"]["unsupported_questions"] == 4
    assert first["outcomes"]["answerable_questions"] == 4
    assert "unsupported_acceptance_count" in first["outcomes"]
    assert check_baseline(first, first) == []
    baseline = {key: first[key] for key in ("schema_version", "configuration", "corpus_hash", "benchmark_hash", "critical")}
    changed = deepcopy(first)
    key = next(key for key, value in first["critical"].items() if value)
    changed["critical"][key] = 0
    assert any("Critical regression" in error for error in check_baseline(changed, baseline))
    changed["benchmark_hash"] = "changed"
    assert any("identity mismatch" in error for error in check_baseline(changed, baseline))
    assert check_baseline(first, {**baseline, "critical": {}})
    changed = deepcopy(first)
    changed["configuration"]["top_k"] = 3
    assert any("configuration mismatch" in error for error in check_baseline(changed, baseline))
