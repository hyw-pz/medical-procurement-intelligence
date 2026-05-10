"""
metrics.py — retrieval evaluation metrics.

Implements Recall@K, MRR, NDCG@K with per-category breakdown.
Designed to surface *where* a retrieval system fails, not just overall scores.

Philosophy: the same as the AML audit — overall numbers are a starting point,
but the actionable insight is always in the breakdown.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    category: str                    # product category of the relevant item
    query_type: str                  # "natural_language" | "abbreviated" | "multilingual" | "spec_only"
    relevant_ids: list[str]          # ground-truth relevant product IDs
    retrieved_ids: list[str]         # ordered list from retriever


@dataclass
class MetricSummary:
    recall_at_k: dict[int, float]    # {1: 0.xx, 5: 0.xx, 10: 0.xx}
    mrr: float
    ndcg_at_k: dict[int, float]
    n_queries: int
    # Breakdowns
    by_category: dict[str, "MetricSummary"] = field(default_factory=dict)
    by_query_type: dict[str, "MetricSummary"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """
    Binary relevance NDCG. Ideal DCG assumes all relevant docs at top positions.
    """
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
        if r in relevant
    )
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

K_VALUES = [1, 5, 10]


def evaluate(results: list[QueryResult]) -> MetricSummary:
    """
    Compute full metric suite over a list of QueryResults.
    Includes per-category and per-query-type breakdowns.
    """
    summary = _compute_metrics(results)
    summary.by_category = _breakdown(results, key="category")
    summary.by_query_type = _breakdown(results, key="query_type")
    return summary


def _compute_metrics(results: list[QueryResult]) -> MetricSummary:
    recalls = {k: [] for k in K_VALUES}
    ndcgs = {k: [] for k in K_VALUES}
    rrs = []

    for r in results:
        relevant = set(r.relevant_ids)
        for k in K_VALUES:
            recalls[k].append(recall_at_k(relevant, r.retrieved_ids, k))
            ndcgs[k].append(ndcg_at_k(relevant, r.retrieved_ids, k))
        rrs.append(reciprocal_rank(relevant, r.retrieved_ids))

    n = len(results)
    return MetricSummary(
        recall_at_k={k: sum(v) / n for k, v in recalls.items()},
        mrr=sum(rrs) / n,
        ndcg_at_k={k: sum(v) / n for k, v in ndcgs.items()},
        n_queries=n,
    )


def _breakdown(
    results: list[QueryResult], key: str
) -> dict[str, MetricSummary]:
    groups: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        groups[getattr(r, key)].append(r)
    return {group: _compute_metrics(items) for group, items in groups.items()}


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

@dataclass
class FailureCase:
    query_id: str
    query_text: str
    category: str
    query_type: str
    relevant_ids: list[str]
    top_retrieved: list[str]        # top-5 retrieved
    recall_at_5: float
    failure_pattern: str            # diagnosed pattern


FAILURE_PATTERNS = {
    "abbreviation": [
        r"\b[A-Z]{2,4}-\d",          # e.g. "PTFE 4-0", "EN455"
        r"\b[A-Z]{2}\d{1,2}\b",
    ],
    "multilingual": [
        r"\b(Stück|Größe|steril|Handschuhe|Einmal|puderfrei)\b",
    ],
    "spec_only": [
        r"^\d+[xX]\d+",              # dimension specs like "10x20"
        r"^\d+\s?(ml|mL|mm|cm)\b",
    ],
}


def diagnose_failures(
    results: list[QueryResult], recall_threshold: float = 0.2
) -> list[FailureCase]:
    """
    Identify queries where recall@5 is below threshold and classify the failure pattern.
    """
    import re

    failures = []
    for r in results:
        relevant = set(r.relevant_ids)
        r5 = recall_at_k(relevant, r.retrieved_ids, 5)
        if r5 <= recall_threshold:
            pattern = _classify_failure(r.query_text)
            failures.append(
                FailureCase(
                    query_id=r.query_id,
                    query_text=r.query_text,
                    category=r.category,
                    query_type=r.query_type,
                    relevant_ids=r.relevant_ids,
                    top_retrieved=r.retrieved_ids[:5],
                    recall_at_5=r5,
                    failure_pattern=pattern,
                )
            )
    return failures


def _classify_failure(query_text: str) -> str:
    import re

    for pattern_name, regexes in FAILURE_PATTERNS.items():
        for regex in regexes:
            if re.search(regex, query_text, re.IGNORECASE):
                return pattern_name
    return "catalog_inconsistency"


# ---------------------------------------------------------------------------
# Pretty-print summary (used in run_eval.py)
# ---------------------------------------------------------------------------

def format_summary(name: str, summary: MetricSummary) -> str:
    lines = [
        f"\n{'='*60}",
        f"  {name}  (n={summary.n_queries})",
        f"{'='*60}",
        f"  Recall@1:  {summary.recall_at_k[1]:.3f}",
        f"  Recall@5:  {summary.recall_at_k[5]:.3f}",
        f"  Recall@10: {summary.recall_at_k[10]:.3f}",
        f"  MRR:       {summary.mrr:.3f}",
        f"  NDCG@10:   {summary.ndcg_at_k[10]:.3f}",
    ]
    if summary.by_category:
        lines.append("\n  By category:")
        for cat, s in sorted(summary.by_category.items()):
            lines.append(f"    {cat:<30} R@5={s.recall_at_k[5]:.3f}  MRR={s.mrr:.3f}")
    if summary.by_query_type:
        lines.append("\n  By query type:")
        for qt, s in sorted(summary.by_query_type.items()):
            lines.append(f"    {qt:<20} R@5={s.recall_at_k[5]:.3f}  MRR={s.mrr:.3f}")
    return "\n".join(lines)
