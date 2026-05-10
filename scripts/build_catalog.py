"""
build_catalog.py — ingest data, build unified catalog, generate eval set, build indexes.

Usage:
    python scripts/build_catalog.py [--config configs/config.yaml] [--no-index]

Steps:
    1. Ingest FDA 510(k) public data (if available) + synthetic supplier catalog
    2. Clean and deduplicate
    3. Write unified catalog to data/processed/catalog.jsonl
    4. Generate ground-truth eval set
    5. Build BM25 + FAISS indexes (unless --no-index)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import yaml


def main(config: dict, build_indexes: bool = True) -> None:
    from src.data.ingest import (
        ingest_fda_510k,
        generate_synthetic_catalog,
        write_catalog,
        load_catalog,
    )
    from src.data.clean import clean_catalog
    from src.data.generate_eval import generate_eval_set, print_eval_stats

    output_dir = Path(config["output_dir"])
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Ingest
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Ingesting product data")
    print("=" * 60)

    products = []

    fda_path = Path("data/raw/fda_510k_sample.csv")
    if fda_path.exists():
        print(f"  Loading FDA 510(k) data from {fda_path}...")
        fda_products = list(ingest_fda_510k(fda_path))
        products.extend(fda_products)
        print(f"  → {len(fda_products)} FDA products loaded")
    else:
        print(f"  FDA data not found at {fda_path} — skipping (synthetic data only)")
        print(f"  Download from: https://www.fda.gov/medical-devices/510k-clearances/downloadable-510k-files")

    print(f"  Generating {config['n_synthetic_products']} synthetic supplier products...")
    synthetic = list(generate_synthetic_catalog(
        n_products=config["n_synthetic_products"],
        seed=config["synthetic_seed"],
    ))
    products.extend(synthetic)
    print(f"  → {len(synthetic)} synthetic products generated")
    print(f"  Total raw: {len(products)} products")

    # ------------------------------------------------------------------
    # Step 2: Clean
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Cleaning and deduplication")
    print("=" * 60)

    products_cleaned = clean_catalog(products)
    print(f"  After cleaning: {len(products_cleaned)} products")

    # ------------------------------------------------------------------
    # Step 3: Write catalog
    # ------------------------------------------------------------------
    catalog_path = Path(config["catalog_path"])
    n_written = write_catalog(iter(products_cleaned), catalog_path)
    print(f"\n  Catalog written to {catalog_path} ({n_written} products)")

    # ------------------------------------------------------------------
    # Step 4: Generate eval set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Generating eval set")
    print("=" * 60)

    catalog = load_catalog(catalog_path)
    eval_path = Path(config["eval_path"])
    eval_items = generate_eval_set(
        catalog=catalog,
        n_queries_per_type=config.get("n_eval_queries", 500) // 16,  # balanced across 4 cats × 4 types
        output_path=eval_path,
    )
    print_eval_stats(eval_items)

    # ------------------------------------------------------------------
    # Step 5: Build indexes
    # ------------------------------------------------------------------
    if not build_indexes:
        print("\nSkipping index build (--no-index)")
        return

    print("\n" + "=" * 60)
    print("Step 5: Building retrieval indexes")
    print("=" * 60)

    from src.retrieval.hybrid_retriever import BM25Retriever, DenseRetriever

    print("  Building BM25 index...")
    bm25 = BM25Retriever(catalog)
    bm25_path = processed_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  → BM25 index saved to {bm25_path}")

    print(f"  Building FAISS index ({config['dense_model']})...")
    dense = DenseRetriever(catalog, model_name=config["dense_model"])
    dense_path = processed_dir / "dense_index.pkl"
    with open(dense_path, "wb") as f:
        pickle.dump(dense, f)
    print(f"  → Dense index saved to {dense_path}")

    # Save catalog index (product_id → product dict) for agent lookups
    catalog_index = {p["product_id"]: p for p in catalog}
    index_path = processed_dir / "catalog_index.json"
    with open(index_path, "w") as f:
        json.dump(catalog_index, f)
    print(f"  → Catalog lookup index saved to {index_path}")

    print("\n" + "=" * 60)
    print("Build complete. Run:")
    print("  python scripts/run_benchmark.py    # evaluate retrievers")
    print("  python scripts/run_agent.py        # interactive procurement agent")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip building FAISS/BM25 indexes (just ingest + eval set)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config, build_indexes=not args.no_index)
