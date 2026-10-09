"""Bounded literal verification; no general semantic entailment."""
import re
from pydantic import BaseModel, Field
from backend.rag_schema import GroundedResponse

class ClaimCheck(BaseModel):
    text: str
    status: str
    reason_code: str
    critical: bool = False
    evidence_ids: list[str] = Field(default_factory=list)

class Verification(BaseModel):
    version: str = "literal-v1"
    status: str = "NOT_VERIFIED"
    claims: list[ClaimCheck] = Field(default_factory=list)
    coverage_complete: bool = False
    blocked: bool = False
    limitation: str = "Literal source support and same-template value checks only; not semantic entailment."

def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip().rstrip(".")

VALUE = re.compile(r"(?<![\w-])\d+(?:[.,:/-]\d+)*(?:\s*%|\b)")

def verify_answer(response: GroundedResponse, operational_text: str = "") -> Verification:
    if response.grounding_status == "Insufficient evidence":
        return Verification(status="NOT_APPLICABLE")
    evidence = [e for e in response.retrieved_evidence if e.source_document in response.sources]
    # Sentence/line units retain unknown headings and summaries as not verified.
    parts = [re.sub(r"^\s*(?:[-*] |\d+[.)] )", "", p).strip()
             for p in re.split(r"(?<=[.!?])\s+|\n+", response.answer) if p.strip()]
    claims = []
    for text in parts[:32]:
        if len(text) > 2000:
            claims.append(ClaimCheck(text=text[:2000], status="NOT_VERIFIED", reason_code="claim_length_bound"))
            continue
        literal = normalize(text)
        source_units = [(e.chunk_id, normalize(re.sub(r"^\s*(?:[-*] |\d+[.)] )", "", unit)))
                        for e in evidence for unit in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", e.text)) if unit.strip()]
        matches = [chunk_id for chunk_id, unit in source_units if literal and literal == unit]
        numeric = bool(VALUE.search(text))
        operational = bool(operational_text and literal in normalize(operational_text))
        if matches:
            claims.append(ClaimCheck(text=text,status="SUPPORTED",reason_code="literal_cited_evidence",evidence_ids=matches))
            continue
        # Existing extractive formatting can omit a leading condition. The
        # identical subject/value suffix confirms its values, but the omitted
        # condition means the full claim remains NOT_VERIFIED, never SUPPORTED.
        conditional_value_match = numeric and any(unit.endswith(literal) and re.match(r"^(?:for|if|when)\b.*?, ", unit) for _, unit in source_units)
        # Contradictions only when the full lexical template matches: this ties
        # values to the same subject/facet, rather than accepting any equal number.
        template = normalize(VALUE.sub("<value>", text))
        contradictory = [chunk_id for chunk_id, unit in source_units if numeric and template == normalize(VALUE.sub("<value>", unit))]
        claims.append(ClaimCheck(text=text,status="UNSUPPORTED" if contradictory else "NOT_VERIFIED",
                                 reason_code="value_mismatch" if contradictory else "conditional_context_not_verified" if conditional_value_match else "outside_literal_coverage",
                                 critical=numeric and bool(evidence) and not operational and not conditional_value_match,evidence_ids=contradictory))
    blocked = any(c.status == "UNSUPPORTED" or (c.critical and c.status != "SUPPORTED") for c in claims)
    complete = len(parts) <= 32 and all(len(p) <= 2000 for p in parts)
    # Truncation may conceal critical unsupported claims: fail closed.
    blocked = blocked or not complete
    status = "UNSUPPORTED" if any(c.status == "UNSUPPORTED" for c in claims) else "NOT_VERIFIED" if not complete or not claims or any(c.status != "SUPPORTED" for c in claims) else "SUPPORTED"
    return Verification(status=status,claims=claims,coverage_complete=complete,blocked=blocked)

def review_signal(question):
    required = bool(re.search(r"\b(?:production|approve|approval|delete|security|credentials|rollback|roll back)\b",question,re.I))
    return {"required":required,"status":"REVIEW_REQUIRED" if required else "NOT_FLAGGED",
            "reason_code":"high_risk_keyword" if required else "no_keyword_match",
            "limitation":"Heuristic signal only; no human review, approval record or queue."}
