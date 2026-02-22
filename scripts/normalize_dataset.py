#!/usr/bin/env python3
"""
Normalize Target handbags CSV/JSON/JSONL files to the canonical schema.

Target columns:
    availability_text, average_rating, best_seller_flag, brand, care_instructions,
    category_breadcrumb, color_options, description, dimensions, dimensions_section,
    image_count, image_url, images_list, is_sale, material_text, materials_section,
    name, new_arrival_flag, price_currency, price_current, price_original,
    product_details, product_id, product_url, promo_excluded, rating_count,
    review_count, scrape_date, size_options, sku, sold_shipped_by

Usage:
    python scripts/normalize_dataset.py <input_file> [<input_file> ...] [--output-dir DIR]
    python scripts/normalize_dataset.py scraper/output/target_handbags_20260222_124633.csv
    python scripts/normalize_dataset.py scraper/output/*.json --output-dir output/normalized
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Canonical output columns (order preserved) ─────────────────────────────
CANONICAL_COLUMNS = [
    "availability_text",
    "average_rating",
    "best_seller_flag",
    "brand",
    "care_instructions",
    "category_breadcrumb",
    "color_options",
    "description",
    "dimensions",
    "dimensions_section",
    "image_count",
    "image_url",
    "images_list",
    "is_sale",
    "material_text",
    "materials_section",
    "name",
    "new_arrival_flag",
    "price_currency",
    "price_current",
    "price_original",
    "product_details",
    "product_id",
    "product_url",
    "promo_excluded",
    "rating_count",
    "review_count",
    "scrape_date",
    "size_options",
    "sku",
    "sold_shipped_by",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _s(v: object) -> str:
    """Safe string coercion; None/empty → ''."""
    if v is None:
        return ""
    s = str(v).strip()
    return s if s not in ("None", "nan") else ""


def _bool_str(v: object) -> str:
    """Normalise truthy values to 'True'/'False' string."""
    if isinstance(v, bool):
        return str(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return "True"
    if s in ("0", "false", "no", ""):
        return "False"
    return s  # leave ambiguous values as-is


def _first_image(images_str: str) -> str:
    """Return the first URL from a pipe-separated images string."""
    if not images_str:
        return ""
    parts = [p.strip() for p in images_str.split("|") if p.strip()]
    return parts[0] if parts else ""


def _image_count(images_str: str) -> str:
    """Count pipe-separated URLs."""
    if not images_str:
        return "0"
    parts = [p.strip() for p in images_str.split("|") if p.strip()]
    return str(len(parts))


def _variant_tcin(row: dict) -> str:
    """
    Extract the true per-variant TCIN (not the parent group ID).

    Priority:
      1. specs JSON field  → {"TCIN": "94150235", ...}
      2. URL preselect param → .../A-94412115?preselect=94150235
      3. Explicit product_id / id field (Schema B)
      4. Parent tcin (Schema A fallback)
    """
    import re

    # 1. specs JSON
    specs_raw = _s(row.get("specs"))
    if specs_raw and specs_raw.startswith("{"):
        try:
            import json
            specs = json.loads(specs_raw)
            tcin_val = _s(specs.get("TCIN") or specs.get("tcin"))
            if tcin_val:
                return tcin_val
        except Exception:
            pass

    # 2. URL preselect param
    url_raw = _s(row.get("url") or row.get("product_url"))
    if url_raw:
        m = re.search(r"[?&]preselect=(\d+)", url_raw)
        if m:
            return m.group(1)

    # 3. Explicit product_id / id field (Schema B)
    pid = _s(row.get("product_id") or row.get("id"))
    if pid:
        return pid

    # 4. Parent tcin fallback
    return _s(row.get("tcin"))


# ── Row normalisation ────────────────────────────────────────────────────────

def normalize_row(row: dict) -> dict:
    """
    Map a raw scraper row (any known schema) to the canonical schema dict.

    Handles two known source schemas:
      Schema A (CSV scraper output): tcin, title, url, brand, breadcrumb,
          price, original_price, is_on_sale, rating, review_count, is_new,
          colors, images, description, feature_bullets, material,
          dimensions_raw, dimensions, care_instructions, upc, in_stock, scraped_at
      Schema B (JSON/earlier CSV): product_id, title/name, url/product_url, ...
    """

    # Strip BOM prefix (\ufeff) from keys — present when CSV saved with UTF-8 BOM
    if any(k.startswith("\ufeff") for k in row):
        row = {k.lstrip("\ufeff"): v for k, v in row.items()}

    # ── product_id ──────────────────────────────────────────────────────────
    # Use variant TCIN (from specs/preselect) to avoid parent-ID collisions
    product_id = _variant_tcin(row)

    # ── name ────────────────────────────────────────────────────────────────
    name = _s(row.get("title") or row.get("name"))

    # ── product_url ─────────────────────────────────────────────────────────
    product_url = _s(row.get("url") or row.get("product_url"))

    # ── brand ────────────────────────────────────────────────────────────────
    brand = _s(row.get("brand"))

    # ── category_breadcrumb ──────────────────────────────────────────────────
    category_breadcrumb = _s(
        row.get("breadcrumb")
        or row.get("category_breadcrumb")
        or row.get("category")
    )

    # ── pricing ─────────────────────────────────────────────────────────────
    price_current = _s(row.get("price") or row.get("price_current"))
    price_original = _s(row.get("original_price") or row.get("price_original"))
    price_currency = _s(row.get("price_currency")) or "USD"

    # ── sale flag ────────────────────────────────────────────────────────────
    is_sale = _bool_str(row.get("is_on_sale") or row.get("is_sale") or "")

    # ── rating / reviews ─────────────────────────────────────────────────────
    average_rating = _s(row.get("rating") or row.get("average_rating"))
    review_count_raw = _s(row.get("review_count"))
    rating_count = _s(row.get("rating_count")) or review_count_raw
    review_count = review_count_raw

    # ── availability ─────────────────────────────────────────────────────────
    in_stock_raw = row.get("in_stock") or row.get("availability_text") or ""
    in_stock_s = str(in_stock_raw).strip().lower()
    if in_stock_s in ("true", "1", "yes", "in stock"):
        availability_text = "In Stock"
    elif in_stock_s in ("false", "0", "no", "out of stock"):
        availability_text = "Out of Stock"
    else:
        availability_text = _s(in_stock_raw)

    # ── images ───────────────────────────────────────────────────────────────
    # Schema A: images = pipe-separated URLs
    # Schema B: images_list already clean; image_url = first URL
    images_list = _s(
        row.get("images")
        or row.get("images_list")
        or row.get("image_list")
    )
    image_url = _s(row.get("image_url")) or _first_image(images_list)
    image_count = _s(row.get("image_count")) or _image_count(images_list)

    # ── color ────────────────────────────────────────────────────────────────
    color_options = _s(row.get("colors") or row.get("color_options") or row.get("color"))

    # ── description ──────────────────────────────────────────────────────────
    description = _s(row.get("description"))

    # ── product_details (feature bullets) ────────────────────────────────────
    product_details = _s(
        row.get("feature_bullets")
        or row.get("product_details")
        or row.get("highlights")
        or row.get("feature_bullet")
    )

    # ── dimensions ───────────────────────────────────────────────────────────
    # dimensions_raw  → human-readable string, e.g. "14 Inches (H) x 21 Inches (W)"
    # dimensions      → JSON-structured dict, e.g. {"H": "14", "W": "21"}
    dimensions = _s(
        row.get("dimensions_raw")        # schema A preferred
        or row.get("dimensions")         # schema B / fallback
    )
    dimensions_section = _s(
        row.get("dimensions")            # JSON field in schema A
        if row.get("dimensions_raw")     # only use JSON when _raw present
        else row.get("dimensions_section")
        or row.get("dimensions")
    )

    # ── materials ────────────────────────────────────────────────────────────
    material_text = _s(
        row.get("material")
        or row.get("material_text")
        or row.get("shell_material")
    )
    materials_section = _s(
        row.get("materials_section")
        or row.get("material_section")
    ) or material_text   # fallback to same value

    # ── care instructions ────────────────────────────────────────────────────
    care_instructions = _s(
        row.get("care_instructions")
        or row.get("care_and_cleaning")
        or row.get("care")
    )

    # ── flags ────────────────────────────────────────────────────────────────
    new_arrival_flag = _bool_str(
        row.get("is_new") or row.get("new_arrival_flag") or ""
    )
    best_seller_flag = _bool_str(
        row.get("best_seller_flag") or row.get("best_seller") or "False"
    )

    # ── sku ──────────────────────────────────────────────────────────────────
    sku = _s(row.get("upc") or row.get("sku"))

    # ── scrape date ──────────────────────────────────────────────────────────
    scrape_date = _s(row.get("scraped_at") or row.get("scrape_date") or row.get("scrape_timestamp"))

    # ── unmapped / always-empty in source ────────────────────────────────────
    promo_excluded = _s(row.get("promo_excluded"))
    size_options = _s(row.get("size_options") or row.get("sizes"))
    sold_shipped_by = _s(row.get("sold_shipped_by") or row.get("sold_by") or row.get("sold_and_shipped_by"))

    return {
        "availability_text": availability_text,
        "average_rating": average_rating,
        "best_seller_flag": best_seller_flag,
        "brand": brand,
        "care_instructions": care_instructions,
        "category_breadcrumb": category_breadcrumb,
        "color_options": color_options,
        "description": description,
        "dimensions": dimensions,
        "dimensions_section": dimensions_section,
        "image_count": image_count,
        "image_url": image_url,
        "images_list": images_list,
        "is_sale": is_sale,
        "material_text": material_text,
        "materials_section": materials_section,
        "name": name,
        "new_arrival_flag": new_arrival_flag,
        "price_currency": price_currency,
        "price_current": price_current,
        "price_original": price_original,
        "product_details": product_details,
        "product_id": product_id,
        "product_url": product_url,
        "promo_excluded": promo_excluded,
        "rating_count": rating_count,
        "review_count": review_count,
        "scrape_date": scrape_date,
        "size_options": size_options,
        "sku": sku,
        "sold_shipped_by": sold_shipped_by,
    }


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # utf-8-sig strips the BOM (\ufeff) that Excel/Windows adds to CSV files
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "products" in data:
            return data["products"]
        return [data]
    if suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported file type: {suffix}")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Target handbags data files to the canonical schema."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="One or more CSV / JSON / JSONL input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files. Defaults to same directory as each input.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="csv",
        help="Output format (default: csv).",
    )
    args = parser.parse_args()

    for input_path in args.inputs:
        input_path = input_path.resolve()
        if not input_path.exists():
            logger.error("File not found: %s", input_path)
            continue

        logger.info("Processing %s …", input_path.name)
        try:
            raw_rows = load_file(input_path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", input_path, exc)
            continue

        normalized = [normalize_row(r) for r in raw_rows]
        logger.info("  %d rows normalized.", len(normalized))

        out_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
        stem = input_path.stem + "_normalized"

        if args.format in ("csv", "both"):
            out_csv = out_dir / (stem + ".csv")
            write_csv(normalized, out_csv)
            logger.info("  → %s", out_csv)

        if args.format in ("json", "both"):
            out_json = out_dir / (stem + ".json")
            write_json(normalized, out_json)
            logger.info("  → %s", out_json)

    logger.info("Done.")


if __name__ == "__main__":
    main()
