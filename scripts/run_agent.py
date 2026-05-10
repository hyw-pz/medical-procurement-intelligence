"""
run_agent.py — interactive procurement agent demo.

Usage:
    # Single query
    python scripts/run_agent.py --query "sterile latex-free examination gloves size M EN455"

    # Interactive REPL
    python scripts/run_agent.py

Requires:
    - Indexes built: python scripts/build_catalog.py
    - ANTHROPIC_API_KEY set in environment
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import yaml


def load_indexes(config: dict):
    processed_dir = Path("data/processed")

    bm25_path = processed_dir / "bm25_index.pkl"
    dense_path = processed_dir / "dense_index.pkl"
    catalog_index_path = processed_dir / "catalog_index.json"

    if not bm25_path.exists() or not dense_path.exists():
        raise FileNotFoundError(
            "Indexes not found. Run: python scripts/build_catalog.py"
        )

    print("Loading indexes...")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(dense_path, "rb") as f:
        dense = pickle.load(f)
    with open(catalog_index_path) as f:
        catalog_index = json.load(f)

    print(f"  {len(catalog_index)} products in catalog index")
    return bm25, dense, catalog_index


def format_brief(brief) -> str:
    lines = [
        "",
        "=" * 60,
        f"PROCUREMENT BRIEF",
        "=" * 60,
        f"Requirement: {brief.requirement}",
        f"Confidence:  {brief.confidence.upper()}",
        "",
        "Shortlisted products:",
    ]

    for i, product in enumerate(brief.shortlisted, 1):
        lines.append(f"  {i}. {product['name']}")
        lines.append(f"     Manufacturer: {product['manufacturer']}")
        lines.append(f"     {product['rationale']}")

    if brief.gaps:
        lines.append("")
        lines.append("⚠ Unmet requirements:")
        for gap in brief.gaps:
            lines.append(f"  - {gap}")

    lines += [
        "",
        "Recommendation:",
        f"  {brief.recommendation}",
        "=" * 60,
    ]
    return "\n".join(lines)


def run_single(agent, query: str) -> None:
    print(f"\nProcessing: {query!r}")
    print("Retrieving candidates...")
    brief = agent.run(query)
    print(format_brief(brief))


def run_repl(agent) -> None:
    print("\nMedical Procurement Agent — interactive mode")
    print("Type a procurement requirement, or 'quit' to exit.\n")

    example_queries = [
        "sterile latex-free examination gloves size M EN455",
        "Einmalspritze 10ml Luer-Lock",
        "foam wound dressing 10x10cm sterile",
        "PTFE 4-0 CV surgical suture",
        "fenestrated sterile surgical drape",
    ]
    print("Example queries:")
    for q in example_queries:
        print(f"  > {q}")
    print()

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        try:
            brief = agent.run(query)
            print(format_brief(brief))
        except Exception as e:
            print(f"Error: {e}")


def main(config: dict, query: str | None) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Export it before running the agent."
        )

    from src.retrieval.hybrid_retriever import BM25Retriever, DenseRetriever, HybridRetriever
    from src.agent.procurement_agent import ProcurementAgent

    bm25, dense, catalog_index = load_indexes(config)
    hybrid = HybridRetriever([bm25, dense], k=config.get("rrf_k", 60))
    agent = ProcurementAgent(retriever=hybrid, catalog_index=catalog_index)

    if query:
        run_single(agent, query)
    else:
        run_repl(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--query",
        default=None,
        help="Single query to process (omit for interactive REPL)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config, args.query)
