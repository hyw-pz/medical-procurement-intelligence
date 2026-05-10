"""
ingest.py — parse and unify product data from heterogeneous sources.

Sources supported:
  - FDA 510(k) public CSV (device name, category, applicant/manufacturer)
  - Synthetic supplier sheets (simulating real-world messy catalog data)

Output: list of dicts with unified schema, written to data/processed/catalog.jsonl
"""

import csv
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import yaml


@dataclass
class Product:
    product_id: str
    name: str
    description: str
    category: str
    manufacturer: str
    certifications: list[str]
    unit: str
    source: str  # "fda_510k" | "synthetic_supplier"


# ---------------------------------------------------------------------------
# FDA 510(k) ingest
# ---------------------------------------------------------------------------

FDA_CATEGORY_MAP = {
    "FRB": "examination gloves",
    "KZG": "surgical gloves",
    "FMF": "face masks",
    "OZO": "surgical drapes",
    "KPR": "syringes",
    "KZE": "catheters",
    "IYO": "infusion sets",
    "MRX": "wound dressings",
}


def ingest_fda_510k(csv_path: Path) -> Iterator[Product]:
    """
    Parse FDA 510(k) clearance CSV.
    Relevant columns: KNUMBER, DEVICENAME, PRODUCTCODE, APPLICANT
    """
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            device_name = row.get("DEVICENAME", "").strip()
            product_code = row.get("PRODUCTCODE", "").strip()
            applicant = row.get("APPLICANT", "").strip()
            k_number = row.get("KNUMBER", "").strip()

            if not device_name or not applicant:
                continue

            category = FDA_CATEGORY_MAP.get(product_code, "medical device")

            yield Product(
                product_id=f"fda_{k_number}",
                name=device_name,
                description=_build_fda_description(device_name, category, applicant),
                category=category,
                manufacturer=applicant,
                certifications=["FDA 510(k)"],
                unit="unit",
                source="fda_510k",
            )


def _build_fda_description(name: str, category: str, manufacturer: str) -> str:
    return f"{name}. Category: {category}. Manufacturer: {manufacturer}. FDA 510(k) cleared."


# ---------------------------------------------------------------------------
# Synthetic supplier catalog ingest
# ---------------------------------------------------------------------------

# Intentional messiness mirrors real supplier data:
#   - inconsistent units (pcs / pieces / Stück)
#   - mixed language entries
#   - abbreviated specs ("PTFE 4-0 CV-8", "latex-free", "EN455 cert.")
#   - same product described differently across suppliers

SYNTHETIC_TEMPLATES = [
    {
        "name": "Latex-Free Examination Gloves {size}, {material}",
        "category": "examination gloves",
        "certifications": ["EN455", "CE"],
        "unit_variants": ["box/100", "box/200", "Karton/100 Stk.", "pcs"],
        "desc_variants": [
            "Powder-free {material} examination gloves, size {size}. EN455 certified. Latex-free.",
            "Untersuchungshandschuhe {material}, Größe {size}, puderfrei. CE-zertifiziert.",
            "{material} gloves s:{size} lf pf EN455",  # abbreviated
        ],
    },
    {
        "name": "Sterile Surgical Drape {type}",
        "category": "surgical drapes",
        "certifications": ["EN13795", "CE"],
        "unit_variants": ["unit", "each", "Stück", "pc"],
        "desc_variants": [
            "Sterile surgical drape, {type} configuration. EN13795 compliant.",
            "Steriles Abdecktuch {type}. Single use.",
            "surg drape {type} sterile EN13795",
        ],
    },
    {
        "name": "Single-Use Syringe {volume}mL",
        "category": "syringes",
        "certifications": ["ISO594", "CE"],
        "unit_variants": ["box/100", "pack/50", "100er-Pack"],
        "desc_variants": [
            "Single-use hypodermic syringe, {volume}mL, luer-lock. ISO594.",
            "Einmalspritze {volume} ml Luer-Lock. CE-konform.",
            "disposable syringe {volume}ml luer-lock iso594 ce",
        ],
    },
    {
        "name": "Wound Dressing {size}cm {type}",
        "category": "wound dressings",
        "certifications": ["CE", "ISO10993"],
        "unit_variants": ["unit", "each", "box/10"],
        "desc_variants": [
            "Sterile wound dressing {size}cm, {type}. CE marked, biocompatible.",
            "Wundverband {size}cm {type}, steril, biokompatibel.",
            "wound dress {size}cm {type} sterile CE iso10993",
        ],
    },
]

SIZES = ["XS", "S", "M", "L", "XL"]
MATERIALS = ["nitrile", "vinyl", "neoprene", "polyisoprene"]
DRAPE_TYPES = ["fenestrated", "full-body", "extremity", "ophthalmic"]
VOLUMES = [1, 2, 5, 10, 20, 50]
WOUND_SIZES = ["5x5", "10x10", "10x20", "15x20"]
WOUND_TYPES = ["foam", "hydrocolloid", "alginate", "silicone"]
SUPPLIER_NAMES = [
    "MedSupply GmbH", "EuroMed AG", "ClinicalPro Ltd", "Sanitech BV",
    "HospitalDirect SE", "MediSource KG", "CareSupply SRL", "TechMed AS",
]


def generate_synthetic_catalog(n_products: int = 6000, seed: int = 42) -> Iterator[Product]:
    """
    Generate synthetic supplier catalog entries with realistic messiness.
    """
    rng = random.Random(seed)

    for i in range(n_products):
        template = rng.choice(SYNTHETIC_TEMPLATES)
        supplier = rng.choice(SUPPLIER_NAMES)
        desc_template = rng.choice(template["desc_variants"])
        unit = rng.choice(template["unit_variants"])

        # Fill in template placeholders
        params = _sample_params(template["category"], rng)
        name = template["name"].format(**params)
        description = desc_template.format(**params)

        yield Product(
            product_id=f"syn_{i:05d}",
            name=name,
            description=description,
            category=template["category"],
            manufacturer=supplier,
            certifications=template["certifications"],
            unit=unit,
            source="synthetic_supplier",
        )


def _sample_params(category: str, rng: random.Random) -> dict:
    if category == "examination gloves":
        return {"size": rng.choice(SIZES), "material": rng.choice(MATERIALS)}
    elif category == "surgical drapes":
        return {"type": rng.choice(DRAPE_TYPES)}
    elif category == "syringes":
        return {"volume": rng.choice(VOLUMES)}
    elif category == "wound dressings":
        return {"size": rng.choice(WOUND_SIZES), "type": rng.choice(WOUND_TYPES)}
    return {}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_catalog(products: Iterator[Product], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for product in products:
            f.write(json.dumps(asdict(product)) + "\n")
            count += 1
    return count


def load_catalog(catalog_path: Path) -> list[dict]:
    with open(catalog_path) as f:
        return [json.loads(line) for line in f if line.strip()]
