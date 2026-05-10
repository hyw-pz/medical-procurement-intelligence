"""
run_benchmark.py — evaluate all retrievers on the ground-truth eval set.

Usage:
    python scripts/run_benchmark.py [--config configs/config.yaml]

Outputs:
    outputs/benchmark_results.csv
    outputs/benchmark_report.html   (opens automatically)
"""

import argparse
import json
import webbrowser
from pathlib import Path

import pandas as pd
import yaml

from src.data.ingest import load_catalog
from src.eval.metrics import QueryResult, evaluate, diagnose_failures, format_summary
from src.retrieval.hybrid_retriever import build_retrievers


def load_eval_set(path: Path) -> list[QueryResult]:
    results = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            results.append(
                QueryResult(
                    query_id=item["query_id"],
                    query_text=item["query_text"],
                    category=item["category"],
                    query_type=item["query_type"],
                    relevant_ids=item["relevant_ids"],
                    retrieved_ids=[],   # filled in per retriever
                )
            )
    return results


def run_benchmark(config: dict) -> None:
    catalog_path = Path(config["catalog_path"])
    eval_path = Path(config["eval_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading catalog from {catalog_path}...")
    catalog = load_catalog(catalog_path)
    print(f"  {len(catalog)} products loaded.")

    print("Loading eval set...")
    eval_queries = load_eval_set(eval_path)
    print(f"  {len(eval_queries)} queries loaded.")

    retrievers = build_retrievers(catalog, config)

    all_rows = []
    summaries = {}

    for retriever_name, retriever in retrievers.items():
        print(f"\nRunning {retriever_name}...")
        query_results = []

        for qr in eval_queries:
            retrieved = retriever.retrieve(qr.query_text, top_k=10)
            query_results.append(
                QueryResult(
                    query_id=qr.query_id,
                    query_text=qr.query_text,
                    category=qr.category,
                    query_type=qr.query_type,
                    relevant_ids=qr.relevant_ids,
                    retrieved_ids=[pid for pid, _ in retrieved],
                )
            )

        summary = evaluate(query_results)
        summaries[retriever_name] = summary
        print(format_summary(retriever_name, summary))

        # collect rows for CSV
        for qr in query_results:
            from src.eval.metrics import recall_at_k, reciprocal_rank, ndcg_at_k
            relevant = set(qr.relevant_ids)
            all_rows.append({
                "retriever": retriever_name,
                "query_id": qr.query_id,
                "category": qr.category,
                "query_type": qr.query_type,
                "recall_at_1": recall_at_k(relevant, qr.retrieved_ids, 1),
                "recall_at_5": recall_at_k(relevant, qr.retrieved_ids, 5),
                "recall_at_10": recall_at_k(relevant, qr.retrieved_ids, 10),
                "mrr": reciprocal_rank(relevant, qr.retrieved_ids),
                "ndcg_at_10": ndcg_at_k(relevant, qr.retrieved_ids, 10),
            })

        # failure analysis for hybrid only
        if retriever_name == "hybrid":
            failures = diagnose_failures(query_results)
            print(f"\nFailure analysis ({len(failures)} queries with recall@5 ≤ 0.2):")
            from collections import Counter
            pattern_counts = Counter(f.failure_pattern for f in failures)
            total = len(failures)
            for pattern, count in pattern_counts.most_common():
                print(f"  {pattern:<30} {count:>4}  ({count/total*100:.1f}%)")

    # save CSV
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # generate HTML report
    report_path = output_dir / "benchmark_report.html"
    from src.report.report_generator import generate_report
    generate_report(report_path, summaries, df, failures)
    print(f"Report saved to {report_path}")
    webbrowser.open(str(report_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_benchmark(config)
