#!/usr/bin/env python3
"""
Normalize and clean Target + Gap product CSVs into a unified schema, then upload to SQLite.

- Maps both sources to a canonical schema with maximum identifiable columns.
- Cleans data to avoid misleading values (UI junk, non-product images, etc.).
- Outputs CSV/JSON and optionally loads to SQLite for querying and embedding pipelines.

Usage:
    python scripts/normalize_and_upload.py data/target_handbags_20260223_014436.csv
    python scripts/normalize_and_upload.py data/*.csv

Default output: output/products_normalized_combined.csv|json, output/products.db
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Collapsed output columns (user-facing schema)
COLLAPSED_COLUMNS = [
    "product_id",
    "source",
    "brand",
    "product_name",
    "leaf_category",
    "price",
    "original_price",
    "discount_percent",
    "is_on_sale",
    "rating",
    "review_count",
    "description",
    "material_text",
    "dimensions",
    "image_url",
    "color_count",
    "timestamp",
    "product_url",
]

# Extended canonical columns for internal normalization
CANONICAL_COLUMNS = [
    "source",
    "product_id",
    "name",
    "brand",
    "product_url",
    "category_breadcrumb",
    "leaf_category",
    "price_currency",
    "price_current",
    "price_original",
    "discount_percent",
    "is_sale",
    "availability_text",
    "average_rating",
    "rating_count",
    "review_count",
    "description",
    "product_details",
    "material_text",
    "materials_section",
    "color_options",
    "size_options",
    "dimensions",
    "dimensions_section",
    "bag_structure",
    "interior_features",
    "exterior_features",
    "closure_type",
    "handle_type",
    "fabric_name",
    "care_instructions",
    "origin",
    "image_url",
    "images_list",
    "image_count",
    "sku",
    "dpci",
    "sold_shipped_by",
    "best_seller_flag",
    "new_arrival_flag",
    "promo_excluded",
    "scrape_date",
]

# Patterns for non-product image URLs (cookie consent, placeholders, etc.)
NON_PRODUCT_IMAGE_PATTERNS = (
    r"cookielaw\.org",
    r"Logo_color_lockup",
    r"baseswatch\?wid=",
)

# UI junk patterns in dimensions (Gap scraped "One Size Sold & shipped by X Return by mail only...")
DIMENSIONS_UI_JUNK = re.compile(
    r"\s*Sold\s*&\s*shipped\s*by\s+[^.]*\.?\s*"
    r"(?:Return by mail only\.?\s*)?"
    r"(?:Only a few left!?\s*)?"
    r"(?:\d+\s*)?"
    r"(?:Add to Bag|Get it before it's gone)?\s*",
    re.IGNORECASE,
)

# Care instructions junk (Gap: "Do not get wet #1174308 Customers also viewed...")
CARE_JUNK_PATTERN = re.compile(
    r"\s*#\d+\s*|Customers also viewed.*|You may also like.*",
    re.IGNORECASE | re.DOTALL,
)

# Material junk (". For every bag sold, STATE donates...")
MATERIAL_JUNK_PATTERNS = (
    r"^\.\s*For every bag sold.*",
    r"^\.\s*$",
)


def _s(v: object) -> str:
    """Safe string coercion; None/empty/nan → ''."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s in ("None", "nan", "{}") else s


def _bool_str(v: object) -> str:
    """Normalize to 'True'/'False' string."""
    if isinstance(v, bool):
        return "True" if v else "False"
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return "True"
    if s in ("0", "false", "no", ""):
        return "False"
    return s


def _first_image(images_str: str) -> str:
    """First product image URL from pipe-separated list (after filtering junk)."""
    urls = _filter_product_images(images_str)
    return urls[0] if urls else ""


def _image_count(images_str: str) -> str:
    """Count product image URLs (excluding junk)."""
    return str(len(_filter_product_images(images_str)))


def _filter_product_images(images_str: str) -> list[str]:
    """Filter out non-product image URLs."""
    if not images_str:
        return []
    urls = [u.strip() for u in images_str.split("|") if u.strip()]
    out = []
    for u in urls:
        if not u.startswith("http"):
            continue
        if any(re.search(p, u) for p in NON_PRODUCT_IMAGE_PATTERNS):
            continue
        out.append(u)
    return out


def _clean_dimensions(val: str) -> str:
    """Remove UI scraped junk from dimensions."""
    if not val:
        return ""
    cleaned = DIMENSIONS_UI_JUNK.sub(" ", val)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # "One Size" is valid; empty after cleanup → keep "One Size" if it was there
    if "One Size" in val and not cleaned:
        return "One Size"
    return cleaned


def _clean_care_instructions(val: str) -> str:
    """Remove 'Customers also viewed' and product ID junk."""
    if not val:
        return ""
    return CARE_JUNK_PATTERN.sub("", val).strip()


def _clean_material(val: str) -> str:
    """Remove non-material text (e.g. STATE donation blurb)."""
    if not val:
        return ""
    for pat in MATERIAL_JUNK_PATTERNS:
        val = re.sub(pat, "", val, flags=re.IGNORECASE).strip()
    if val in (".", ".."):
        return ""
    return val


def _clean_product_details(val: str, description: str) -> str:
    """Remove product_details when it's just a fragment of description or junk."""
    if not val:
        return ""
    # If it's a leading fragment of description, drop it (redundant)
    if description and val.strip().startswith("."):
        val = val.lstrip(".").strip()
    if len(val) < 20 and description and val in description:
        return ""
    return val


def _price_str(val: object) -> str:
    """Format price; use empty string for 0 when it means 'no original price'."""
    if val is None:
        return ""
    s = str(val).strip()
    if s in ("", "None", "nan"):
        return ""
    try:
        f = float(s)
        if f == 0:
            return ""  # Avoid misleading 0 for price_original
        return s
    except ValueError:
        return s


def _extract_material_from_description(desc: str) -> str:
    """Heuristic: extract material mentions from description."""
    if not desc or len(desc) < 20:
        return ""
    desc_lower = desc.lower()
    materials = []
    # Common material phrases
    for phrase in [
        "vegan leather",
        "genuine leather",
        "faux leather",
        "recycled nylon",
        "nylon",
        "polyester",
        "cotton",
        "polyurethane",
        "acrylic",
        "canvas",
        "suede",
        "recycled materials",
    ]:
        if phrase in desc_lower:
            materials.append(phrase.title())
    return ", ".join(dict.fromkeys(materials)) if materials else ""


def _detect_source(row: dict) -> str:
    """Detect source from row structure."""
    if "tcin" in row or "scraped_at" in row or ("url" in row and "target.com" in str(row.get("url", ""))):
        return "target"
    if "product_id" in row and "product_url" in row and "gap.com" in str(row.get("product_url", "")):
        return "gap"
    if "availability_text" in row and "category_breadcrumb" in row:
        return "gap"  # Canonical Gap export
    return "target"  # Default


def _variant_tcin(row: dict) -> str:
    """Extract per-variant TCIN from Target row."""
    specs_raw = _s(row.get("specs"))
    if specs_raw and specs_raw.startswith("{"):
        try:
            spec = json.loads(specs_raw)
            tcin = _s(spec.get("TCIN") or spec.get("tcin"))
            if tcin:
                return tcin
        except Exception:
            pass
    url_raw = _s(row.get("url") or row.get("product_url"))
    if url_raw:
        m = re.search(r"[?&]preselect=(\d+)", url_raw)
        if m:
            return m.group(1)
    return _s(row.get("product_id") or row.get("tcin") or row.get("id"))


def normalize_target_row(row: dict) -> dict:
    """Normalize Target scraper row to canonical schema."""
    if any(k.startswith("\ufeff") for k in row):
        row = {k.lstrip("\ufeff"): v for k, v in row.items()}

    product_id = _variant_tcin(row)
    name = _s(row.get("title") or row.get("name"))
    product_url = _s(row.get("url") or row.get("product_url"))
    brand = _s(row.get("brand"))
    category_breadcrumb = _s(row.get("breadcrumb") or row.get("category_breadcrumb"))
    leaf_category = _s(row.get("leaf_category"))

    price_current = _s(row.get("price") or row.get("price_current"))
    price_orig_raw = row.get("original_price") or row.get("price_original")
    price_original = _price_str(price_orig_raw) if price_orig_raw not in (None, "") else ""
    if not price_original and _s(price_orig_raw) in ("0", "0.0"):
        price_original = ""
    discount = _s(row.get("discount_percent"))
    price_currency = _s(row.get("price_currency")) or "USD"
    is_sale = _bool_str(row.get("is_on_sale") or row.get("is_sale"))

    in_stock = str(row.get("in_stock", "")).strip().lower()
    if in_stock in ("true", "1", "yes", "in stock"):
        availability_text = "In Stock"
    elif in_stock in ("false", "0", "no"):
        availability_text = "Out of Stock"
    else:
        availability_text = "In Stock" if in_stock else ""

    avg_rating = _s(row.get("rating") or row.get("average_rating"))
    review_count = _s(row.get("review_count"))
    rating_count = _s(row.get("rating_count")) or review_count

    description = _s(row.get("description"))
    product_details = _s(row.get("feature_bullets") or row.get("product_details"))

    material_text = _s(row.get("material") or row.get("material_text") or row.get("shell_material"))
    materials_section = _s(row.get("materials_section")) or material_text

    color_options = _s(row.get("colors") or row.get("color_options"))
    size_options = _s(row.get("size_options") or row.get("sizes"))

    dimensions_raw = _s(row.get("dimensions_raw"))
    dimensions_json = _s(row.get("dimensions"))
    if dimensions_raw:
        dimensions = dimensions_raw
        dimensions_section = dimensions_json if dimensions_json.startswith("{") else ""
    else:
        dimensions = dimensions_json
        dimensions_section = ""

    bag_structure = _s(row.get("bag_structure"))
    interior_features = _s(row.get("interior_features"))
    exterior_features = _s(row.get("exterior_features"))
    closure_type = _s(row.get("closure_type"))
    handle_type = _s(row.get("handle_type"))
    fabric_name = _s(row.get("fabric_name"))
    care_instructions = _s(row.get("care_instructions") or row.get("care_and_cleaning"))
    origin = _s(row.get("origin"))

    images_list = _s(row.get("images") or row.get("images_list"))
    image_url = _s(row.get("image_url")) or _first_image(images_list)
    image_count = _s(row.get("image_count")) or _image_count(images_list)

    sku = _s(row.get("upc") or row.get("sku"))
    dpci = _s(row.get("dpci"))
    sold_shipped_by = _s(row.get("sold_shipped_by"))
    best_seller_flag = _bool_str(row.get("best_seller_flag") or "False")
    new_arrival_flag = _bool_str(row.get("is_new") or row.get("new_arrival_flag"))
    promo_excluded = _s(row.get("promo_excluded"))
    scrape_date = _s(row.get("scraped_at") or row.get("scrape_date"))

    return {
        "source": "target",
        "product_id": product_id,
        "name": name,
        "brand": brand,
        "product_url": product_url,
        "category_breadcrumb": category_breadcrumb,
        "leaf_category": leaf_category,
        "price_currency": price_currency,
        "price_current": price_current,
        "price_original": price_original,
        "discount_percent": discount,
        "is_sale": is_sale,
        "availability_text": availability_text,
        "average_rating": avg_rating,
        "rating_count": rating_count,
        "review_count": review_count,
        "description": description,
        "product_details": product_details,
        "material_text": material_text,
        "materials_section": materials_section,
        "color_options": color_options,
        "size_options": size_options,
        "dimensions": dimensions,
        "dimensions_section": dimensions_section,
        "bag_structure": bag_structure,
        "interior_features": interior_features,
        "exterior_features": exterior_features,
        "closure_type": closure_type,
        "handle_type": handle_type,
        "fabric_name": fabric_name,
        "care_instructions": care_instructions,
        "origin": origin,
        "image_url": image_url,
        "images_list": images_list,
        "image_count": image_count,
        "sku": sku,
        "dpci": dpci,
        "sold_shipped_by": sold_shipped_by,
        "best_seller_flag": best_seller_flag,
        "new_arrival_flag": new_arrival_flag,
        "promo_excluded": promo_excluded,
        "scrape_date": scrape_date,
    }


def normalize_gap_row(row: dict) -> dict:
    """Normalize Gap export row (already canonical) and map extra fields."""
    if any(k.startswith("\ufeff") for k in row):
        row = {k.lstrip("\ufeff"): v for k, v in row.items()}

    def g(k: str, *alt: str) -> str:
        for key in (k,) + alt:
            v = row.get(key)
            if v is not None and str(v).strip() not in ("", "nan", "None"):
                return _s(v)
        return ""

    product_id = g("product_id", "id")
    price_curr = g("price_current", "price")
    price_orig = row.get("price_original") or row.get("price_original")
    price_original = _price_str(price_orig)
    if not price_original and str(price_orig or "").strip() in ("0", "0.0"):
        price_original = ""

    images_list = g("images_list", "images")
    image_url = g("image_url") or _first_image(images_list)
    image_count = g("image_count") or _image_count(images_list)

    return {
        "source": "gap",
        "product_id": product_id,
        "name": g("name", "title"),
        "brand": g("brand"),
        "product_url": g("product_url", "url"),
        "category_breadcrumb": g("category_breadcrumb", "category"),
        "leaf_category": "",  # Gap doesn't have leaf
        "price_currency": g("price_currency") or "USD",
        "price_current": price_curr,
        "price_original": price_original,
        "discount_percent": "",
        "is_sale": _bool_str(row.get("is_sale")),
        "availability_text": g("availability_text"),
        "average_rating": g("average_rating", "rating"),
        "rating_count": g("rating_count"),
        "review_count": g("review_count"),
        "description": g("description"),
        "product_details": g("product_details"),
        "material_text": g("material_text", "material"),
        "materials_section": g("materials_section"),
        "color_options": g("color_options", "colors"),
        "size_options": g("size_options", "sizes"),
        "dimensions": g("dimensions"),
        "dimensions_section": g("dimensions_section"),
        "bag_structure": "",
        "interior_features": "",
        "exterior_features": "",
        "closure_type": "",
        "handle_type": "",
        "fabric_name": "",
        "care_instructions": g("care_instructions"),
        "origin": "",
        "image_url": image_url,
        "images_list": images_list,
        "image_count": image_count,
        "sku": g("sku"),
        "dpci": "",
        "sold_shipped_by": g("sold_shipped_by"),
        "best_seller_flag": _bool_str(row.get("best_seller_flag")),
        "new_arrival_flag": _bool_str(row.get("new_arrival_flag")),
        "promo_excluded": g("promo_excluded"),
        "scrape_date": g("scrape_date"),
    }


def clean_row(row: dict) -> dict:
    """Clean row to avoid misleading data."""
    out = dict(row)

    # Dimensions: remove UI junk
    for key in ("dimensions", "dimensions_section"):
        if out.get(key):
            cleaned = _clean_dimensions(out[key])
            if cleaned != out[key]:
                out[key] = cleaned
            # dimensions_section: if it contains same junk as dimensions, use dimensions
            if key == "dimensions_section" and cleaned and "Sold" in str(out.get("dimensions", "")):
                pass  # already cleaned

    # Images: filter non-product URLs
    img_list = out.get("images_list")
    if img_list:
        filtered = _filter_product_images(img_list)
        out["images_list"] = " | ".join(filtered)
        out["image_url"] = filtered[0] if filtered else ""
        out["image_count"] = str(len(filtered))

    # Price original: empty when 0 (misleading)
    if out.get("price_original") in ("0", "0.0"):
        out["price_original"] = ""

    # Material: remove junk
    for key in ("material_text", "materials_section"):
        if out.get(key):
            out[key] = _clean_material(out[key])

    # Extract material from description if missing
    if not out.get("material_text") and out.get("description"):
        extracted = _extract_material_from_description(out["description"])
        if extracted:
            out["material_text"] = extracted
            if not out.get("materials_section"):
                out["materials_section"] = extracted

    # Product details: remove redundant fragments
    out["product_details"] = _clean_product_details(
        out.get("product_details", ""), out.get("description", "")
    )

    # Care instructions: remove junk
    if out.get("care_instructions"):
        out["care_instructions"] = _clean_care_instructions(out["care_instructions"])

    return out


def _count_colors(color_options: str) -> str:
    """Parse color_options (e.g. 'Black | Brown | Pink') and return count as string."""
    if not color_options or not str(color_options).strip():
        return ""
    # Split by pipe or comma; strip each; filter out empty and " - Out of Stock" suffixes
    parts = re.split(r"[|,]", str(color_options))
    colors = []
    for p in parts:
        p = re.sub(r"\s*-\s*Out of Stock.*$", "", p, flags=re.IGNORECASE).strip()
        if p and not p.lower().startswith("size"):
            colors.append(p)
    return str(len(colors)) if colors else ""


def collapse_row(row: dict) -> dict:
    """Collapse canonical row to user-facing schema."""
    desc = _s(row.get("description"))
    details = _s(row.get("product_details"))
    description = desc if desc else details
    if desc and details and desc != details:
        description = f"{desc}; {details}"

    # Append bag structure and features for richer similarity search embeddings
    extra = []
    for label, val in [
        ("Bag structure", row.get("bag_structure")),
        ("Interior features", row.get("interior_features")),
        ("Exterior features", row.get("exterior_features")),
    ]:
        v = _s(val)
        if v:
            extra.append(f"{label}: {v}")
    if extra:
        description = f"{description}; {'; '.join(extra)}" if description else "; ".join(extra)

    orig = row.get("price_original")
    original_price = str(orig).strip() if orig and str(orig).strip() not in ("", "0", "0.0") else ""
    discount_percent = _s(row.get("discount_percent"))
    is_on_sale = _bool_str(row.get("is_on_sale") or row.get("is_sale"))

    dims = _s(row.get("dimensions")) or _s(row.get("dimensions_section"))
    if dims and dims.startswith("{"):
        try:
            d = json.loads(dims)
            dims = ", ".join(f"{k}: {v}" for k, v in d.items()) if isinstance(d, dict) else dims
        except Exception:
            pass

    return {
        "product_id": _s(row.get("product_id")),
        "source": _s(row.get("source")) or "target",
        "brand": _s(row.get("brand")),
        "product_name": _s(row.get("name")),
        "leaf_category": _s(row.get("leaf_category")),
        "price": _s(row.get("price_current")),
        "original_price": original_price,
        "discount_percent": discount_percent,
        "is_on_sale": is_on_sale,
        "rating": _s(row.get("average_rating")),
        "review_count": _s(row.get("review_count")),
        "description": description,
        "material_text": _s(row.get("material_text")) or _s(row.get("materials_section")),
        "dimensions": dims,
        "image_url": _s(row.get("image_url")),
        "color_count": _count_colors(row.get("color_options") or ""),
        "timestamp": _s(row.get("scrape_date")),
        "product_url": _s(row.get("product_url")),
    }


def load_csv(path: Path) -> list[dict]:
    """Load CSV with UTF-8-sig."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
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
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported: {suffix}")


def write_output(rows: list[dict], out_dir: Path, stem: str, fmt: str) -> None:
    collapsed = [collapse_row(r) for r in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("csv", "both"):
        path = out_dir / f"{stem}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLLAPSED_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(collapsed)
        logger.info("  → %s", path)
    if fmt in ("json", "both"):
        path = out_dir / f"{stem}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(collapsed, f, ensure_ascii=False, indent=2)
        logger.info("  → %s", path)


def upload_to_sqlite(rows: list[dict], db_path: Path, table: str = "products") -> None:
    """Upload collapsed rows to SQLite."""
    try:
        import sqlite3
    except ImportError:
        logger.warning("sqlite3 not available; skipping DB upload.")
        return

    collapsed = [collapse_row(r) for r in rows]
    if not collapsed:
        logger.warning("No rows to upload.")
        return

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols = COLLAPSED_COLUMNS
    col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
        conn.execute(f'CREATE UNIQUE INDEX idx_{table}_url ON "{table}" (product_url)')

        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(f'"{c}"' for c in cols)
        for r in collapsed:
            vals = [str(r.get(c, "") or "")[:10000] for c in cols]
            conn.execute(
                f'INSERT OR REPLACE INTO "{table}" ({col_list}) VALUES ({placeholders})',
                vals,
            )
        conn.commit()
        logger.info("  → SQLite %s (%d rows)", db_path, len(collapsed))
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and clean Target + Gap product CSVs; output and optionally upload to SQLite."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="Target and/or Gap CSV/JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output).",
    )
    parser.add_argument(
        "--output-name",
        default="products_normalized_combined",
        help="Output file stem (default: products_normalized_combined).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("output/products.db"),
        help="SQLite database path (default: output/products.db).",
    )
    parser.add_argument(
        "--table",
        default="products",
        help="SQLite table name (default: products).",
    )
    parser.add_argument(
        "--require-description",
        action="store_true",
        help="Filter to only rows with description (default: keep all rows).",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for inp in args.inputs:
        inp = Path(inp).resolve()
        if not inp.exists():
            logger.error("File not found: %s", inp)
            continue

        logger.info("Processing %s …", inp.name)
        try:
            raw = load_file(inp)
        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

        source = _detect_source(raw[0]) if raw else "target"
        normalize_fn = normalize_gap_row if source == "gap" else normalize_target_row

        for r in raw:
            row = normalize_fn(r)
            row = clean_row(row)
            pid = row.get("product_id", "")
            key = (row.get("source", source), pid)
            if not pid or key in seen:
                continue
            seen.add(key)
            all_rows.append(row)

        logger.info("  %d rows from %s", len(raw), source)

    if not all_rows:
        logger.error("No rows to write.")
        return

    # Optionally filter to rows with description
    if args.require_description:
        before = len(all_rows)
        all_rows = [r for r in all_rows if (r.get("description") or "").strip()]
        if len(all_rows) < before:
            logger.info("Filtered to %d rows with description (removed %d).", len(all_rows), before - len(all_rows))
        if not all_rows:
            logger.error("No rows with description to write.")
            return

    logger.info("Total %d unique rows.", len(all_rows))
    write_output(all_rows, args.output_dir, args.output_name, args.format)

    if args.db:
        upload_to_sqlite(all_rows, args.db, args.table)

    logger.info("Done.")


if __name__ == "__main__":
    main()
