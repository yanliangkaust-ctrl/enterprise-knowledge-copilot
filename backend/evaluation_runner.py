"""Local Phase 2A harness. Run: python -m backend.evaluation_runner --check.

No shared store, network generation, or persisted session state is used.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from backend.agentic_orchestrator import ask_agentic
from backend.answer_generation import evidence_items_from_results
from backend.document_ingestion import chunk_documents, load_markdown_documents, parse_markdown_content
from backend.graph_builder import build_knowledge_graph
from backend.rag_pipeline import retrieve_chunks
from backend.retrieval_evaluation import metrics_for_results, validate_questions
from backend.semantic_retrieval import LocalEmbeddingModel, VectorIndex

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "data" / "evaluation"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def resolve_selector(selector, chunks):
    matches = [chunk for chunk in chunks if chunk.document_name == selector.get("document")
               and chunk.section_heading == selector.get("section")
               and text_hash(chunk.text) == selector.get("text_sha256")]
    if len(matches) != 1:
        raise ValueError(f"Unresolved or ambiguous gold evidence: {selector}")
    return matches[0].chunk_id


def load_fixture():
    original = json.loads((EVALUATION / "ground_truth_questions.json").read_text(encoding="utf-8"))
    supplemental = json.loads((EVALUATION / "phase2_supplemental.json").read_text(encoding="utf-8"))
    validate_questions(original)
    validate_questions(supplemental["questions"])
    if len(original) != 10 or supplemental.get("schema_version") != 1:
        raise ValueError("Unexpected benchmark schema or original benchmark size")
    if {q["id"] for q in original} & {q["id"] for q in supplemental["questions"]}:
        raise ValueError("Benchmark IDs overlap")
    documents = load_markdown_documents(ROOT / "data" / "sample_docs")
    documents += [parse_markdown_content(d["name"], d["text"]) for d in supplemental["extra_documents"]]
    names = [d.name for d in documents]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate corpus document")
    chunks = chunk_documents(documents)
    for q in original + supplemental["questions"]:
        if not set(q["expected_sources"]) <= set(names):
            raise ValueError(f"Unknown gold source: {q['id']}")
    for q in supplemental["questions"]:
        if type(q.get("answerable")) is not bool or bool(q["expected_sources"]) != q["answerable"]:
            raise ValueError(f"Invalid answerability: {q['id']}")
        if {s["document"] for s in q["gold_evidence"]} != set(q["expected_sources"]):
            raise ValueError(f"Gold evidence/source mismatch: {q['id']}")
        selectors = q["gold_evidence"] + q.get("partial_evidence", [])
        for fact in q["required_facts"]:
            if not fact.get("any_of") or any(not isinstance(s, str) or not s.strip() for s in fact["any_of"]):
                raise ValueError(f"Invalid fact label: {q['id']}")
            if not fact.get("evidence") or any(s not in q["gold_evidence"] for s in fact["evidence"]):
                raise ValueError(f"Fact lacks gold support: {q['id']}")
            selectors += fact["evidence"]
        for selector in selectors:
            resolve_selector(selector, chunks)
    return original, supplemental, documents, chunks


def integrity_errors(evidence, sources, chunks):
    by_id = {c.chunk_id: c for c in chunks}
    errors = []
    for item in evidence:
        chunk = by_id.get(item.chunk_id)
        if chunk is None or (item.source_document, item.section, item.text) != (
                chunk.document_name, chunk.section_heading, chunk.text):
            errors.append(f"Evidence integrity failure: {item.chunk_id}")
    retrieved_names = {item.source_document for item in evidence}
    if not set(sources) <= retrieved_names:
        errors.append("Fabricated source reference")
    return errors


def relevance_metrics(evidence, gold_ids):
    metrics = {}
    for k in (1, 3, 5):
        found = len({e.chunk_id for e in evidence[:k]} & gold_ids)
        metrics[f"Evidence Precision@{k}"] = found / k
        metrics[f"Evidence Recall@{k}"] = found / len(gold_ids) if gold_ids else 0.0
    return metrics


def outcome(q, result, chunks):
    response = result.response
    refused = response.grounding_status == "Insufficient evidence"
    accepted = not refused
    facts = []
    answer = " ".join(response.answer.casefold().split())
    for fact in q["required_facts"]:
        present = accepted and any(" ".join(s.casefold().split()) in answer for s in fact["any_of"])
        support_ids = {resolve_selector(s, chunks) for s in fact["evidence"]}
        cited = any(e.chunk_id in support_ids and e.source_document in response.sources
                    for e in response.retrieved_evidence)
        facts.append({"alternatives": fact["any_of"], "present": present, "source_supported": present and cited})
    return {"accepted": accepted, "refused": refused,
            "evidence_sufficient": result.trace.evidence_sufficient,
            "unsupported_acceptance": not q["answerable"] and accepted,
            "false_refusal": q["answerable"] and refused,
            "required_fact_coverage": sum(f["present"] for f in facts) / len(facts) if facts else None,
            "source_supported_fact_coverage": sum(f["source_supported"] for f in facts) / len(facts) if facts else None,
            "facts": facts, "answer": response.answer, "sources": response.sources,
            "route": [d.tool for d in result.trace.tool_decisions],
            "sufficiency_reason": result.trace.sufficiency_reason}


def average(rows):
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def run_evaluation():
    original, supplemental, documents, chunks = load_fixture()
    index = VectorIndex(LocalEmbeddingModel(use_sentence_transformer=False))
    index.build(chunks)
    graph = build_knowledge_graph(chunks)
    report = {"schema_version": 1,
              "configuration": {"backend": index.embedding_model.backend, "dimensions": 384,
                                "generation": "Demo / Local", "top_k": 5, "sessions": "fresh/no context",
                                "keyword_unit": "chunk", "precision_denominator": "K; unique documents",
                                "evidence_precision_denominator": "K; unique gold chunks",
                                "ranking": "one Top-5 call per query/strategy; metrics slice frozen result slots",
                                "python": platform.python_version(),
                                "dependencies": {name: version(name) for name in ("numpy", "networkx", "langgraph", "pydantic")}},
              "corpus_hash": digest([{"name": d.name, "text": d.raw_text} for d in documents]),
              "benchmark_hash": digest({"original": original, "supplemental": supplemental}),
              "corpus": {"documents": len(documents), "chunks": len(chunks)},
              "queries": [], "integrity_errors": [], "critical": {}}
    modes = {"keyword": "Keyword", "vector_hash": "Semantic", "graph": "Graph", "hybrid": "Hybrid"}
    supplemental_ids = {q["id"] for q in supplemental["questions"]}
    for q in original + supplemental["questions"]:
        is_supplemental = q["id"] in supplemental_ids
        gold = {resolve_selector(s, chunks) for s in q.get("gold_evidence", [])}
        partial = {resolve_selector(s, chunks) for s in q.get("partial_evidence", [])}
        row = {"id": q["id"], "question": q["question"], "question_type": q["question_type"],
               "suite": "supplemental" if is_supplemental else "original",
               "answerable": q.get("answerable", True), "expected_sources": q["expected_sources"], "strategies": {}}
        for strategy in (*modes, "agentic"):
            if strategy == "agentic":
                # Each invocation has no session context and uses a freshly compiled graph.
                result = ask_agentic(q["question"], chunks, index, graph, generation_mode="Demo / Local", top_k=5)
                evidence = result.response.retrieved_evidence
                sources = result.response.sources
            else:
                results = retrieve_chunks(q["question"], chunks, index, graph, modes[strategy], 5)
                evidence = evidence_items_from_results(results)
                sources = []
            errors = integrity_errors(evidence, sources, chunks)
            report["integrity_errors"] += [f"{q['id']}/{strategy}: {error}" for error in errors]
            entry = {"metrics": metrics_for_results(q["expected_sources"], evidence),
                     "retrieved": [{**e.model_dump(exclude={"text"}), "text_sha256": text_hash(e.text), "relevance": (
                         "gold" if e.chunk_id in gold else "partial" if e.chunk_id in partial else
                         "non-gold" if is_supplemental else "document-only/unjudged"),
                         "document_relevant": e.source_document in q["expected_sources"]} for e in evidence],
                     "integrity_errors": errors}
            if is_supplemental:
                entry["evidence_metrics"] = relevance_metrics(evidence, gold)
                if strategy == "agentic":
                    entry["outcome"] = outcome(q, result, chunks)
            elif strategy == "agentic":
                entry["outcome"] = {"accepted": result.trace.evidence_sufficient,
                                    "refused": result.response.grounding_status == "Insufficient evidence",
                                    "answer": result.response.answer, "sources": sources}
            row["strategies"][strategy] = entry
        for critical in q.get("critical", []):
            entry = row["strategies"][critical["strategy"]]
            value = entry.get("outcome", {}).get(critical["field"], entry["metrics"].get(critical["field"]))
            if not isinstance(value, (bool, int, float)):
                raise ValueError(f"Invalid critical field: {critical}")
            report["critical"][f"{q['id']}/{critical['strategy']}/{critical['field']}"] = value
        report["queries"].append(row)
    report["aggregates"] = {}
    for suite in ("original", "supplemental"):
        rows = [r for r in report["queries"] if r["suite"] == suite and r["answerable"]]
        report["aggregates"][suite] = {strategy: {"positive_questions": len(rows),
            **average([r["strategies"][strategy]["metrics"] for r in rows]),
            **(average([r["strategies"][strategy]["evidence_metrics"] for r in rows]) if suite == "supplemental" else {})}
            for strategy in (*modes, "agentic")}
    rows = [r for r in report["queries"] if r["suite"] == "supplemental"]
    outcomes = [r["strategies"]["agentic"]["outcome"] for r in rows]
    negative_count = sum(not r["answerable"] for r in rows)
    positive_count = len(rows) - negative_count
    facts = [fact for o in outcomes for fact in o["facts"]]
    report["outcomes"] = {
        "path": "agentic; supplemental only", "unsupported_questions": negative_count,
        "unsupported_acceptance_count": sum(o["unsupported_acceptance"] for o in outcomes),
        "unsupported_acceptance_rate": sum(o["unsupported_acceptance"] for o in outcomes) / negative_count if negative_count else None,
        "answerable_questions": positive_count,
        "false_refusal_count": sum(o["false_refusal"] for o in outcomes),
        "false_refusal_rate": sum(o["false_refusal"] for o in outcomes) / positive_count if positive_count else None,
        "required_fact_count": len(facts),
        "required_fact_coverage": sum(f["present"] for f in facts) / len(facts) if facts else None,
        "source_supported_fact_coverage": sum(f["source_supported"] for f in facts) / len(facts) if facts else None}
    return report


def check_baseline(report, baseline):
    errors = []
    for key in ("schema_version", "corpus_hash", "benchmark_hash"):
        if report[key] != baseline.get(key):
            errors.append(f"Baseline identity mismatch: {key}; explicit review/rebaseline required")
    for key, value in report["configuration"].items():
        if key not in {"python", "dependencies"} and baseline.get("configuration", {}).get(key) != value:
            errors.append(f"Baseline configuration mismatch: {key}")
    critical = baseline.get("critical")
    if not critical or set(critical) != set(report["critical"]):
        errors.append("Missing or changed critical-case contract")
    else:
        for key, expected in critical.items():
            if report["critical"][key] < expected:
                errors.append(f"Critical regression: {key}: {report['critical'][key]} < {expected}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVALUATION / "phase2_report.json")
    parser.add_argument("--baseline", type=Path, default=EVALUATION / "phase2_baseline.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        report = run_evaluation()
        repeated = run_evaluation()
        errors = list(report["integrity_errors"])
        if report != repeated:
            errors.append("Nondeterministic repeated local evaluation")
        if args.check:
            errors += check_baseline(report, json.loads(args.baseline.read_text(encoding="utf-8")))
        report["gate"] = {"passed": not errors, "errors": errors, "repeat_equal": report == repeated,
                          "baseline_checked": args.check}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(args.output), "gate": report["gate"], "outcomes": report["outcomes"]}))
        return 1 if errors else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        failure = {"gate": {"passed": False, "errors": [str(exc)]}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
