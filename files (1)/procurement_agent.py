"""
procurement_agent.py — LLM-powered procurement recommendation agent.

Flow:
  1. Receive free-text procurement requirement
  2. Retrieve top-K candidates (hybrid retriever)
  3. LLM re-ranks candidates and filters out poor matches
  4. LLM generates a structured sourcing brief

This is intentionally a simple, transparent agent — no complex tool-calling
chains. The goal is to show that retrieval quality gates everything downstream:
a good brief is impossible with bad candidates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

from src.retrieval.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class SourcingBrief:
    requirement: str
    shortlisted: list[dict]      # [{product_id, name, manufacturer, rationale}]
    recommendation: str          # 1-2 sentence plain-language recommendation
    gaps: list[str]              # requirements not met by any candidate
    confidence: str              # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RERANK_SYSTEM = """
You are a medical procurement specialist. You will be given a procurement requirement
and a list of candidate products retrieved from a supplier catalog.

Your task:
1. Identify which candidates genuinely match the requirement (product type, specs, certifications).
2. Rank the matches from best to worst fit.
3. Flag any requirements that no candidate satisfies.

Respond ONLY with a JSON object — no preamble, no markdown fences:
{
  "shortlisted": [
    {
      "product_id": "...",
      "name": "...",
      "manufacturer": "...",
      "rationale": "1-2 sentences explaining why this matches"
    }
  ],
  "gaps": ["list of unmet requirements"],
  "confidence": "high | medium | low"
}
""".strip()

BRIEF_SYSTEM = """
You are a medical procurement specialist writing a sourcing brief for a hospital buyer.
Be concise, specific, and clinically precise. Avoid generic filler.
""".strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProcurementAgent:
    def __init__(self, retriever: HybridRetriever, catalog_index: dict[str, dict]):
        """
        catalog_index: product_id → product dict, for fast lookup after retrieval.
        """
        self.retriever = retriever
        self.catalog_index = catalog_index
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    def run(self, requirement: str, top_k: int = 10) -> SourcingBrief:
        # Step 1: retrieve candidates
        candidates = self.retriever.retrieve(requirement, top_k=top_k)
        candidate_products = [
            self.catalog_index[pid]
            for pid, _ in candidates
            if pid in self.catalog_index
        ]

        # Step 2: LLM re-rank and filter
        rerank_result = self._rerank(requirement, candidate_products)

        # Step 3: generate sourcing brief
        recommendation = self._generate_brief(requirement, rerank_result)

        return SourcingBrief(
            requirement=requirement,
            shortlisted=rerank_result.get("shortlisted", []),
            recommendation=recommendation,
            gaps=rerank_result.get("gaps", []),
            confidence=rerank_result.get("confidence", "low"),
        )

    def _rerank(self, requirement: str, candidates: list[dict]) -> dict:
        candidate_text = json.dumps(
            [
                {
                    "product_id": p["product_id"],
                    "name": p["name"],
                    "description": p["description"],
                    "certifications": p["certifications"],
                    "manufacturer": p["manufacturer"],
                }
                for p in candidates
            ],
            indent=2,
        )

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=RERANK_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Requirement: {requirement}\n\nCandidates:\n{candidate_text}",
                }
            ],
        )

        try:
            return json.loads(message.content[0].text)
        except (json.JSONDecodeError, IndexError, KeyError):
            return {"shortlisted": [], "gaps": ["LLM response parsing failed"], "confidence": "low"}

    def _generate_brief(self, requirement: str, rerank_result: dict) -> str:
        shortlisted = rerank_result.get("shortlisted", [])
        gaps = rerank_result.get("gaps", [])

        if not shortlisted:
            return "No suitable products found in the current catalog for this requirement."

        top = shortlisted[0]
        gap_note = f" Note: the following requirements were not met: {', '.join(gaps)}." if gaps else ""

        prompt = (
            f"Write a 2-sentence sourcing recommendation for a hospital buyer.\n"
            f"Requirement: {requirement}\n"
            f"Best match: {top['name']} by {top['manufacturer']} — {top['rationale']}\n"
            f"Total shortlisted: {len(shortlisted)} products.{gap_note}\n"
            f"Be specific and clinical. Do not use filler phrases."
        )

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=BRIEF_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
