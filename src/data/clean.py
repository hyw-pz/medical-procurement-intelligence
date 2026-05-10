"""
clean.py — normalise and deduplicate the unified product catalog.

Key transformations:
  - Normalise unit strings (Stück → unit, Karton/100 Stk. → box/100)
  - Strip boilerplate phrases that add noise to retrieval
  - Normalise whitespace and punctuation
  - Deduplicate on (name_normalised, manufacturer) to remove near-duplicates
    across sources
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Unit normalisation map (German → standard)
# ---------------------------------------------------------------------------

UNIT_MAP = {
    r"stück": "unit",
    r"stk\.?": "unit",
    r"karton/(\d+)\s*stk\.?": r"box/\1",
    r"(\d+)er[\s-]pack": r"pack/\1",
    r"pack/(\d+)": r"pack/\1",
    r"box/(\d+)": r"box/\1",
    r"pcs": "unit",
    r"pieces?": "unit",
    r"each": "unit",
    r"pc\.?": "unit",
}

# Phrases that appear frequently but contribute no retrieval signal
BOILERPLATE = [
    r"single use",
    r"single-use",
    r"einmalgebrauch",
    r"manufactured by [\w\s]+\.",
    r"ce-konform\.",
    r"ce conformant\.",
]


def normalise_unit(unit: str) -> str:
    u = unit.lower().strip()
    for pattern, replacement in UNIT_MAP.items():
        u = re.sub(pattern, replacement, u)
    return u


def normalise_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_description(desc: str) -> str:
    """Remove boilerplate phrases from product descriptions."""
    cleaned = desc
    for pattern in BOILERPLATE:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Remove trailing punctuation left by deletions
    cleaned = re.sub(r"[,\.\s]+$", "", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def dedup_key(product: dict) -> str:
    """
    Key for identifying near-duplicate products.
    Two products are considered duplicates if they share normalised name + manufacturer.
    """
    name_norm = normalise_text(product["name"])
    manufacturer_norm = normalise_text(product["manufacturer"])
    return f"{name_norm}||{manufacturer_norm}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_catalog(products: list) -> list[dict]:
    """
    Accept either a list of Product dataclasses or dicts.
    Returns a list of cleaned dicts.
    """
    # Normalise to dicts
    dicts = []
    for p in products:
        dicts.append(asdict(p) if hasattr(p, "__dataclass_fields__") else dict(p))

    cleaned = []
    seen_keys: set[str] = set()
    stats = {"unit_normalised": 0, "desc_cleaned": 0, "duplicates_removed": 0}

    for product in dicts:
        # Normalise unit
        raw_unit = product.get("unit", "unit")
        normed_unit = normalise_unit(raw_unit)
        if normed_unit != raw_unit:
            stats["unit_normalised"] += 1
        product["unit"] = normed_unit

        # Clean description
        raw_desc = product.get("description", "")
        cleaned_desc = clean_description(raw_desc)
        if cleaned_desc != raw_desc:
            stats["desc_cleaned"] += 1
        product["description"] = cleaned_desc

        # Deduplicate
        key = dedup_key(product)
        if key in seen_keys:
            stats["duplicates_removed"] += 1
            continue
        seen_keys.add(key)

        cleaned.append(product)

    print(f"  Cleaning stats: {stats}")
    return cleaned
