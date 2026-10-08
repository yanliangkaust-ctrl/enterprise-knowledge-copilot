"""Bounded lexical guardrail for requested factual facets, not entailment.

Require a value/assignment statement about the requested subject. Topic overlap,
headings, unknown-value statements, and facts about another subject do not suffice.
Unrecognized question forms retain the existing gate behavior.
"""
import re
from collections.abc import Sequence

from backend.rag_schema import EvidenceItem
from backend.observability import emit, timed


_FACET_WORDS = {
    "price", "cost", "pricing", "fee", "budget", "protocol", "date", "time",
    "quantity", "number", "value", "configuration", "setting", "configured",
    "owner", "ownership", "responsible", "team", "location", "located",
    "many", "much", "often", "long", "when", "where",
}
_QUESTION_WORDS = {
    "what", "which", "who", "how", "is", "are", "the", "a", "an", "of",
    "for", "does", "do", "did", "its", "their", "it", "that", "this",
    "use", "uses", "used", "own", "owns", "to", "be", "on", "in", "at", "and", "was", "were",
}
_UNKNOWN = re.compile(
    r"\b(?:unknown|unspecified|undetermined|unavailable|tbd|not\s+(?:known|specified|documented|defined|available)|"
    r"no\s+(?:documented|specified|known)|has\s+no|does\s+not|doesn't|cannot|can't)\b", re.I)
_NUMBER = r"\b\d+(?:[.,]\d+)*\b"
_DURATION = rf"{_NUMBER}\s*(?:milliseconds?|ms|seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
_DATE = r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}:\d{2}(?:\s*[ap]m)?)\b"
_CONFIG_KEYS = {"port", "version", "timeout", "limit", "threshold", "interval", "setting", "configuration", "configured", "value"}


def _words(text):
    # Only normalize common inflections; this is deliberately not a semantic model.
    return {word[:-1] if word.endswith("s") and len(word) > 3 else word
            for word in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.casefold())}


def requested_facets(question: str) -> list[tuple[str, set[str]]]:
    """Identify narrow fact requests per clause, with lexical subject anchors."""
    requests = []
    for clause in re.split(r"\band\s+(?=(?:what|which|who|where|when|how)\b)", question.casefold()):
        tokens = set(re.findall(r"[a-z]+", clause))
        facets = []
        if tokens & {"price", "cost", "pricing", "fee", "budget"}:
            facets.append("price/cost")
        if "protocol" in tokens:
            facets.append("protocol")
        if "when" in tokens or re.search(r"\b(?:what|which)\s+(?:is\s+the\s+)?date\b", clause):
            facets.append("date/time")
        if re.search(r"\bhow\s+(?:often|long)\b|\bwhat\s+time\b", clause):
            facets.append("duration/frequency")
        if re.search(r"\b(?:owns?|owner|ownership|responsible)\b", clause):
            facets.append("ownership")
        if "where" in tokens or tokens & {"location", "located"}:
            facets.append("location")
        if re.search(r"\bhow\s+(?:many|much)\b|\b(?:quantity|number\s+of)\b", clause) and "price/cost" not in facets:
            facets.append("quantity")
        if tokens & _CONFIG_KEYS and re.search(r"\b(?:what|which|how)\b", clause):
            facets.append("configuration/value")
        # In common subject-verb forms, exclude the requested operation from the
        # subject: "what protocol does X use ...", "where does X write ...".
        subject = re.search(r"\b(?:does|do|did)\s+(.+?)\s+(?:use|run|write|store|replicate|cost|have|contain|retain|listen|wait)\b", clause)
        anchors = _words(subject.group(1) if subject else clause) - _words(" ".join(_QUESTION_WORDS | _FACET_WORDS | _CONFIG_KEYS))
        if re.search(r"\b(?:that|this|it|its|their)\b", clause) and anchors <= {"risk"}:
            # A deictic follow-up has no standalone subject. Existing session and
            # graph checks remain in charge; still require an actual assignment.
            anchors = set()
        for facet in facets:
            scoped_anchors = set(anchors)
            if facet == "configuration/value":
                scoped_anchors |= _words(" ".join(tokens & (_CONFIG_KEYS - {"configuration", "configured", "setting", "value"})))
            if facet == "quantity" and subject:
                scoped_anchors |= _words(clause[:subject.start()]) - _words(" ".join(_QUESTION_WORDS | _FACET_WORDS))
            requests.append((facet, scoped_anchors))
    return requests


def _has_value(facet, statement):
    if _UNKNOWN.search(statement) or "?" in statement:
        return False
    if facet == "price/cost":
        return bool(re.search(r"\b(?:price|costs?|pricing|fee|budget)\b", statement) and
                    re.search(r"(?:[$€£]\s*\d|\b\d+(?:[.,]\d+)*\s*(?:usd|eur|gbp|dollars?|euros?|pounds?)\b|\bfree\b|\b(?:price|cost|fee|budget)\s*(?:is|:|=)\s*\d)", statement))
    if facet == "protocol":
        return bool(re.search(r"\b(?:uses?|supports?|implements?|speaks?|runs?\s+over)\s+(?:the\s+)?[\w-]+(?:\s+[\w-]+){0,4}\s+protocol\b|\bprotocol\s*(?:is|:|=)\s*(?!used\b|required\b)[\w-]+", statement))
    if facet == "date/time":
        return bool(re.search(_DATE, statement) or re.search(r"\b(?:date|year)\s*(?:is|:|=)\s*\d{4}\b", statement))
    if facet == "duration/frequency":
        return bool(re.search(_DURATION, statement))
    if facet == "ownership":
        return bool(re.search(r"\b(?:owns?|owned\s+by|operated\s+by|managed\s+by|responsible\s+for|owner\s*:)\s*[a-z][\w-]*", statement))
    if facet == "location":
        return bool(re.search(r"\b(?:located|hosted|deployed|stored|runs?|writes?|stores?|resides?)\s+(?:\w+\s+){0,5}(?:in|on|at|to)\s+(?:the\s+)?[a-z][\w-]*", statement))
    if facet == "quantity":
        return bool(re.search(_NUMBER, statement) and re.search(r"\b(?:has|have|contains?|holds?|retains?|allows?|supports?|count|quantity|number|maximum|minimum)\b", statement))
    if facet == "configuration/value":
        return bool(re.search(r"\b(?:port|version|timeout|limit|threshold|interval|setting|configuration|value)\s*(?:is|:|=|of|to|at)?\s*(?:\d[\w.-]*|true|false|enabled|disabled)\b|\b(?:configured|set)\s+to\s+[\w.-]+", statement))
    return False


def evidence_supports_facet(facet: str, anchors: set[str], evidence: Sequence[EvidenceItem]) -> bool:
    """Use the same subject/value predicate for selection and the refusal gate.

Section context binds labeled values (e.g. Owner:) to their subject. A different
subject's statement elsewhere in the chunk cannot supply the missing facet.
"""
    for item in evidence:
        text = item.text.replace("**", "")
        statements = re.split(r"(?<=[.!?])\s+|\n(?=\s*(?:[-*]|[A-Z]))", text)
        for statement in statements:
            normalized = " ".join(statement.casefold().split())
            scope = _words(normalized)
            if re.match(r"\s*[-*]?\s*(?:owner|protocol|price|cost|date|location|port|version|timeout|limit|threshold|interval|value)\s*[:=]", normalized):
                scope |= _words(item.section)
            if anchors <= scope and _has_value(facet, normalized):
                return True
    return False


@timed("factual_support")
def unsupported_facets(question: str, evidence: Sequence[EvidenceItem]) -> list[str]:
    """Require support for each recognized subject/facet pair."""
    missing = []
    for facet, anchors in requested_facets(question):
        supported = evidence_supports_facet(facet, anchors, evidence)
        emit("factual_support_check", facet=facet, supported=supported,
             reason_code="fact_supported" if supported else "missing_factual_support")
        if not supported and facet not in missing:
            missing.append(facet)
    return missing


def explicit_ownership_support(question: str, evidence: Sequence[EvidenceItem]) -> bool:
    """Text-only ownership policy; does not change the shared facet predicate.

Single subjects require a scoped assignment. Coordinated responsibility questions
may draw support from multiple explicit named-team action statements, provided
every requested subject token is covered by such statements. Topic mentions and
unknown-owner labels receive no credit. This is lexical support, not entailment.
"""
    requests = [(facet, anchors) for facet, anchors in requested_facets(question) if facet == "ownership"]
    if not requests:
        return False
    coordinated = bool(re.search(r"\band\b", question, re.I))
    for facet, anchors in requests:
        if not anchors:
            # An unresolved pronoun is not a subject. Text-only fallback cannot
            # infer an owner from an arbitrary assignment in the retrieved set.
            return False
        covered = set()
        supported = False
        for item in evidence:
            for statement in re.split(r"(?<=[.!?])\s+|\n(?=\s*(?:[-*]|[A-Z]))", item.text.replace("**", "")):
                normalized = " ".join(statement.casefold().split())
                if _UNKNOWN.search(normalized) or re.search(r"\b(?:unassigned|no owner|not responsible)\b", normalized) or "?" in normalized:
                    continue
                assignment = _has_value("ownership", normalized)
                team_action = bool(re.search(
                    r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,5}\s+Team\s+"
                    r"(?:owns|manages|operates|maintains|reviews|approves|patches|responds\s+to)\s+\S", statement))
                if not (assignment or team_action):
                    continue
                scope = _words(normalized)
                if re.match(r"\s*[-*]?\s*owner\s*:", normalized):
                    scope |= _words(item.section)
                if anchors <= scope:
                    supported = True
                covered |= anchors & scope
        if not supported and not (coordinated and anchors and anchors <= covered):
            return False
    return True
