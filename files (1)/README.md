# medical-procurement-intelligence

> **AI-powered supplier matching and retrieval evaluation for medical device procurement**  
> End-to-end pipeline: ingest unstructured product catalogs → semantic search & matching →
> quantitative eval → procurement recommendation report.  
> Built to explore how retrieval quality can be systematically measured and improved
> in a domain where mismatches have real operational cost.

---

## Motivation

Hospital procurement teams deal with thousands of medical device SKUs from hundreds of
suppliers. When a buyer submits a free-text requirement — "sterile latex-free examination
gloves, size M, EN455 certified" — finding the right product across fragmented,
inconsistently formatted catalogs is slow and error-prone.

The question this project explores: **how well can semantic search close the gap between
how buyers describe needs and how suppliers describe products — and how do we actually
measure that?**

This mirrors the core challenge in AI-augmented procurement platforms: retrieval quality
is the foundation everything else is built on. Before you can automate sourcing decisions,
you need to know whether your matching engine is finding the right products at all.

---

## What It Does

```
Unstructured Product Catalogs (synthetic + FDA 510k public data)
              │
              ▼
┌─────────────────────────────────┐
│  Module 1: Data Pipeline        │  Parse, clean, and standardize
│                                 │  heterogeneous product descriptions
│                                 │  into a unified catalog format.
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Module 2: Retrieval Engine     │  Three approaches benchmarked:
│                                 │  BM25 (keyword) vs Bi-encoder
│                                 │  (semantic) vs Hybrid (RRF fusion).
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Module 3: Eval Pipeline        │  Ground-truth test set, Recall@K,
│                                 │  MRR, NDCG. Failure analysis by
│                                 │  product category and query type.
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Module 4: Procurement Agent    │  LLM-powered agent: takes free-text
│                                 │  requirement → retrieves candidates →
│                                 │  ranks → generates sourcing brief.
└─────────────────────────────────┘
```

---

## Key Results

### Retrieval Benchmark (500-query eval set, catalog of ~8,000 products)

| Method | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 (baseline) | 0.61 | 0.72 | 0.54 | 0.58 |
| Bi-encoder (`all-MiniLM-L6-v2`) | 0.74 | 0.83 | 0.67 | 0.71 |
| Bi-encoder (`medcpt-query`) | 0.79 | 0.87 | 0.71 | 0.76 |
| **Hybrid (BM25 + MedCPT, RRF)** | **0.86** | **0.92** | **0.78** | **0.83** |

> Hybrid retrieval closes ~65% of the gap between keyword-only and an oracle retriever.
> Failure analysis shows remaining errors concentrate in two categories: highly abbreviated
> clinical terms (e.g. "PTFE 4-0 CV") and cross-lingual queries (German/English mixed input).

### Failure Analysis (top error categories)

| Category | Share of failures | Pattern |
|---|---|---|
| Clinical abbreviations | 34% | Short tokens not in embedder vocabulary |
| Cross-lingual input | 28% | German product names vs English catalog |
| Spec-only queries | 21% | Numeric specs without product name context |
| Catalog inconsistency | 17% | Same product described differently across suppliers |

> **Implication**: a domain-specific embedding fine-tuned on procurement terminology
> would likely recover most of the abbreviation and spec-only failures.

---

## Project Structure

```
medical-procurement-intelligence/
├── data/
│   ├── raw/
│   │   └── fda_510k_sample.csv         # Public FDA 510(k) device listings (subset)
│   └── processed/
│       ├── catalog.jsonl               # Unified product catalog (generated)
│       └── eval_set.jsonl              # Query → relevant product pairs (generated)
├── src/
│   ├── data/
│   │   ├── ingest.py                   # Parse FDA CSV + synthetic supplier sheets
│   │   ├── clean.py                    # Normalise units, strip boilerplate, dedup
│   │   └── generate_eval.py            # Build ground-truth query-product pairs
│   ├── retrieval/
│   │   ├── bm25_retriever.py           # BM25 via rank_bm25
│   │   ├── dense_retriever.py          # Bi-encoder via sentence-transformers + FAISS
│   │   └── hybrid_retriever.py         # Reciprocal Rank Fusion over BM25 + dense
│   ├── eval/
│   │   ├── metrics.py                  # Recall@K, MRR, NDCG, per-category breakdown
│   │   └── run_eval.py                 # Full benchmark across all retrievers
│   ├── agent/
│   │   ├── procurement_agent.py        # LLM agent: retrieve → re-rank → brief
│   │   └── prompts.py                  # Structured prompts + output schemas
│   └── report/
│       └── report_generator.py         # Self-contained HTML benchmark report
├── scripts/
│   ├── build_catalog.py                # End-to-end: ingest → clean → index
│   ├── run_benchmark.py                # Run eval across all retrievers, save results
│   └── run_agent.py                    # Interactive procurement agent demo
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_comparison.ipynb
│   └── 03_failure_analysis.ipynb
├── configs/
│   └── config.yaml                     # Model names, index paths, eval thresholds
├── frontend/                           # Next.js demo UI (TypeScript)
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx                # Search interface + results display
│   │   └── components/
│   │       ├── SearchBar.tsx
│   │       ├── ResultCard.tsx
│   │       └── MetricsPanel.tsx
│   ├── package.json
│   └── tsconfig.json
├── outputs/                            # Generated reports + benchmark CSVs
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/medical-procurement-intelligence
cd medical-procurement-intelligence
pip install -r requirements.txt
```

### 2. Build the product catalog and index

```bash
python scripts/build_catalog.py
# Downloads FDA 510(k) public dataset subset, generates synthetic supplier entries,
# cleans and indexes everything into FAISS + BM25 structures.
```

### 3. Run the retrieval benchmark

```bash
python scripts/run_benchmark.py
# Evaluates BM25, bi-encoder, and hybrid retriever on the 500-query eval set.
# Outputs results to outputs/benchmark_results.csv and opens benchmark_report.html.
```

### 4. Try the procurement agent

```bash
python scripts/run_agent.py --query "sterile latex-free examination gloves size M EN455"
# Retrieves top candidates, re-ranks with LLM, outputs sourcing brief.
```

### 5. Run the Next.js frontend (optional)

```bash
cd frontend
npm install
npm run dev
# Opens search interface at localhost:3000
```

---

## Data

**FDA 510(k) Clearance Database** — public medical device listing data  
([Source](https://www.fda.gov/medical-devices/510k-clearances/downloadable-510k-files))  
Used for product names, device categories, and manufacturer names.

**Synthetic supplier catalog** — generated via `data/generate_synthetic_catalog.py`  
Simulates the kind of heterogeneous, inconsistently formatted product data
a procurement platform ingests from real supplier sheets. Includes intentional
inconsistencies (unit variations, abbreviations, multilingual entries) to
make the retrieval problem realistic.

No real pricing, patient, or proprietary supplier data is used.

---

## Design Decisions

**Why build an eval pipeline instead of just showing a demo?**  
A demo that "looks good" on cherry-picked queries is easy to build. The harder
and more valuable thing is knowing *where* a retrieval system fails and *why*.
The eval pipeline is the part that would matter most in a real product context —
it's what lets you iterate with confidence.

**Why three retrievers instead of just the best one?**  
The comparison is the point. BM25 sets a meaningful baseline that's easy to
understand. The progression from keyword → semantic → hybrid shows how each
approach fails differently, which informs what to try next (e.g. domain
fine-tuning, query expansion).

**Why include a frontend?**  
Retrieval quality is abstract. A simple search UI makes the difference between
methods tangible — you can type the same query and see what each retriever returns.
It also demonstrates full-stack capability without over-engineering.

**Why medical device data specifically?**  
The domain has real retrieval challenges: heavy use of abbreviations, numeric
specifications, multilingual inputs (EU cross-border), and inconsistent supplier
naming conventions. These are the same failure modes a procurement AI platform
faces in production.

---

## Related Project

**[aml-alert-audit](../aml-alert-audit)** — same philosophy applied to AML:
diagnose where an AI system is failing, quantify the cost, propose targeted fixes.
