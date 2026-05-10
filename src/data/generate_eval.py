"""
generate_eval.py — build a ground-truth query → relevant product eval set.

Strategy:
  For each product in the catalog, generate queries of four types:
    1. natural_language  — fluent buyer description
    2. abbreviated       — clinical shorthand (e.g. "nitrile gloves M lf pf EN455")
    3. multilingual      — German/English mixed input (common in EU procurement)
    4. spec_only         — numeric specs without product name context

  Relevant IDs = all products in the same category + same key attributes.
  We sample a fixed number of queries per type to keep the eval set balanced.

Output: data/processed/eval_set.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Query templates per category and type
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "examination gloves": {
        "natural_language": [
            "latex-free examination gloves size {size} powder-free",
            "disposable {material} gloves for clinical examination, size {size}",
            "sterile-free {material} exam gloves {size} EN455 certified",
            "non-sterile examination gloves {size} latex free {material}",
        ],
        "abbreviated": [
            "{material} gloves {size} lf pf",
            "exam gloves {size} {material} EN455",
            "{size} {material} lf exam glv",
            "gloves {material} {size} lf pf EN455 cert",
        ],
        "multilingual": [
            "Untersuchungshandschuhe {size} {material} puderfrei",
            "latexfreie Handschuhe {size} {material}",
            "{material} Untersuchungshandschuhe Größe {size} lf",
            "Handschuhe {material} {size} pf CE-zertifiziert",
        ],
        "spec_only": [
            "{size} {material} lf",
            "EN455 {size} pf",
            "{material} {size}",
            "lf pf {size} {material}",
        ],
    },
    "surgical drapes": {
        "natural_language": [
            "sterile surgical drape {type} single use EN13795",
            "{type} sterile drape for surgical procedures",
            "disposable sterile {type} drape",
            "surgical field drape {type} sterile",
        ],
        "abbreviated": [
            "surg drape {type} sterile",
            "{type} drape EN13795",
            "sterile {type} drape su",
            "drape {type} st EN13795",
        ],
        "multilingual": [
            "steriles Abdecktuch {type} Einmalgebrauch",
            "{type} chirurgisches Abdecktuch steril",
            "OP-Abdecktuch {type} EN13795",
            "sterile Abdeckung {type} su",
        ],
        "spec_only": [
            "{type} EN13795",
            "sterile {type}",
            "{type} single use",
            "{type} su sterile",
        ],
    },
    "syringes": {
        "natural_language": [
            "single use syringe {volume}mL luer lock",
            "disposable hypodermic syringe {volume} ml",
            "{volume}mL luer-lock syringe ISO594",
            "single-use {volume}mL syringe sterile",
        ],
        "abbreviated": [
            "syringe {volume}ml ll",
            "{volume}ml syr luer-lock iso594",
            "disp syr {volume}ml",
            "{volume}ml su syringe ISO594",
        ],
        "multilingual": [
            "Einmalspritze {volume}ml Luer-Lock",
            "{volume} ml Spritze Einmalgebrauch",
            "Einmalspritze {volume}ml steril ISO594",
            "{volume}ml Spritze ll CE",
        ],
        "spec_only": [
            "{volume}ml luer-lock",
            "{volume}mL ISO594",
            "{volume} ml ll su",
            "ISO594 {volume}ml",
        ],
    },
    "wound dressings": {
        "natural_language": [
            "sterile {type} wound dressing {size}cm",
            "{size}cm {type} wound dressing sterile single use",
            "biocompatible sterile {type} dressing {size}",
            "wound dressing {size}cm {type} CE marked",
        ],
        "abbreviated": [
            "wound dress {size}cm {type} st",
            "{type} dress {size} CE iso10993",
            "{size}cm {type} wd sterile",
            "WD {size} {type} CE",
        ],
        "multilingual": [
            "Wundverband {size}cm {type} steril",
            "{type} Wundauflage {size} biokompatibel",
            "steriler {type} Verband {size}cm",
            "{size}cm {type} Wundverband CE",
        ],
        "spec_only": [
            "{size}cm {type}",
            "{type} {size} sterile",
            "{size} {type} CE iso10993",
            "{type} {size}cm su",
        ],
    },
}

# Attribute values matching those used in ingest.py
ATTR_VALUES: dict[str, dict[str, list]] = {
    "examination gloves": {"size": ["XS", "S", "M", "L", "XL"], "material": ["nitrile", "vinyl", "neoprene", "polyisoprene"]},
    "surgical drapes": {"type": ["fenestrated", "full-body", "extremity", "ophthalmic"]},
    "syringes": {"volume": [1, 2, 5, 10, 20, 50]},
    "wound dressings": {"size": ["5x5", "10x10", "10x20", "15x20"], "type": ["foam", "hydrocolloid", "alginate", "silicone"]},
}


# ---------------------------------------------------------------------------
# Ground truth matching
# ---------------------------------------------------------------------------

def _extract_attrs(product: dict) -> dict[str, str]:
    """Pull attribute values from product name/description via simple parsing."""
    attrs: dict[str, str] = {}
    name = product["name"].lower()
    desc = product["description"].lower()

    # size
    for s in ["xs", "s", "m", "l", "xl"]:
        if re.search(rf"\b{s}\b", name) or re.search(rf"\bsize {s}\b", desc):
            attrs["size"] = s.upper()
            break

    # material
    for mat in ["nitrile", "vinyl", "neoprene", "polyisoprene"]:
        if mat in name or mat in desc:
            attrs["material"] = mat
            break

    # drape type
    for dtype in ["fenestrated", "full-body", "extremity", "ophthalmic"]:
        if dtype in name or dtype in desc:
            attrs["type"] = dtype
            break

    # volume
    vol_match = re.search(r"(\d+)\s*m[lL]", name)
    if vol_match:
        attrs["volume"] = vol_match.group(1)

    # wound size
    size_match = re.search(r"(\d+[xX]\d+)cm", name)
    if size_match:
        attrs["size"] = size_match.group(1)

    # wound type
    for wtype in ["foam", "hydrocolloid", "alginate", "silicone"]:
        if wtype in name or wtype in desc:
            attrs["type"] = wtype
            break

    return attrs


def find_relevant_ids(query_attrs: dict, category: str, catalog: list[dict]) -> list[str]:
    """
    Find all products in the same category that share key attributes with the query.
    Liberal matching: a product is relevant if category matches and at least one
    key attribute matches (or the product has no parseable attributes).
    """
    relevant = []
    for product in catalog:
        if product["category"] != category:
            continue
        product_attrs = _extract_attrs(product)
        if not product_attrs:
            continue
        # check overlap on shared keys
        shared_keys = set(query_attrs) & set(product_attrs)
        if not shared_keys:
            relevant.append(product["product_id"])
            continue
        if all(product_attrs.get(k) == v for k, v in query_attrs.items() if k in product_attrs):
            relevant.append(product["product_id"])
    return relevant


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_eval_set(
    catalog: list[dict],
    n_queries_per_type: int = 30,
    seed: int = 42,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Generate a balanced eval set with equal coverage across categories and query types.

    Total queries = n_categories × n_query_types × n_queries_per_type
                  = 4 × 4 × 30 = 480 by default (close to the 500 cited in README).
    """
    rng = random.Random(seed)
    eval_items = []
    query_counter = 0

    for category, type_templates in TEMPLATES.items():
        attr_space = ATTR_VALUES.get(category, {})

        for query_type, templates in type_templates.items():
            for _ in range(n_queries_per_type):
                # sample random attribute values
                attrs = {k: str(rng.choice(v)) for k, v in attr_space.items()}

                # fill template
                template = rng.choice(templates)
                try:
                    query_text = template.format(**attrs)
                except KeyError:
                    continue

                # find relevant products
                relevant_ids = find_relevant_ids(attrs, category, catalog)

                if not relevant_ids:
                    continue   # skip if no relevant products found (shouldn't happen)

                query_counter += 1
                eval_items.append({
                    "query_id": f"q_{query_counter:04d}",
                    "query_text": query_text,
                    "category": category,
                    "query_type": query_type,
                    "relevant_ids": relevant_ids,
                    "attrs": attrs,
                })

    rng.shuffle(eval_items)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in eval_items:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(eval_items)} eval queries to {output_path}")

    return eval_items


# ---------------------------------------------------------------------------
# Quick stats
# ---------------------------------------------------------------------------

def print_eval_stats(eval_items: list[dict]) -> None:
    from collections import Counter

    print(f"\nEval set: {len(eval_items)} queries")

    cat_counts = Counter(item["category"] for item in eval_items)
    print("\nBy category:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<30} {n}")

    type_counts = Counter(item["query_type"] for item in eval_items)
    print("\nBy query type:")
    for qt, n in sorted(type_counts.items()):
        print(f"  {qt:<20} {n}")

    avg_relevant = sum(len(item["relevant_ids"]) for item in eval_items) / len(eval_items)
    print(f"\nAvg relevant products per query: {avg_relevant:.1f}")
