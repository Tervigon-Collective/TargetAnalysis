#!/usr/bin/env python3
"""
Use LLM to fill missing product descriptions using other available fields.

Loads retail CSVs (Target, Gap, J.Crew, Aritzia, Abercrombie), detects schema,
and generates descriptions for rows where description is empty or very short.
Uses name, brand, category, material, dimensions, feature_bullets, product_details, etc.

Usage:
    python scripts/fill_descriptions_llm.py data/target_bags_1.csv data/gap_bag*.csv
    python scripts/fill_descriptions_llm.py data/*.csv -o output/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _s(v) -> str:
    if v is None or (isinstance(v, float) and (v != v or v == float("nan"))):
        return ""
    return str(v).strip()


def _detect_source(row: dict, path: Path) -> str:
    """Detect retailer from row content or path."""
    url = _s(row.get("url") or row.get("product_url"))
    if "target.com" in url:
        return "target"
    if "gap.com" in url:
        return "gap"
    if "jcrew.com" in url:
        return "jcrew"
    if "aritzia.com" in url:
        return "aritzia"
    if "abercrombie.com" in url:
        return "abercrombie"
    name = (path.name or "").lower()
    if "target" in name:
        return "target"
    if "gap" in name:
        return "gap"
    if "jcrew" in name:
        return "jcrew"
    if "aritzia" in name:
        return "aritzia"
    if "abercrombie" in name:
        return "abercrombie"
    if "tcin" in row:
        return "target"
    return "unknown"


def _gather_context(row: dict, source: str) -> dict:
    """Gather all useful fields for LLM context, normalized across schemas."""
    ctx = {}
    # Name/title
    ctx["name"] = _s(row.get("name") or row.get("title") or row.get("product_name"))
    ctx["brand"] = _s(row.get("brand"))
    # Category
    ctx["category"] = _s(row.get("category_breadcrumb") or row.get("breadcrumb") or row.get("leaf_category") or row.get("category"))
    ctx["leaf_category"] = _s(row.get("leaf_category") or row.get("category"))
    # Description-like
    ctx["description"] = _s(row.get("description"))
    ctx["feature_bullets"] = _s(row.get("feature_bullets"))
    ctx["product_details"] = _s(row.get("product_details"))
    # Specs
    ctx["material"] = _s(row.get("material_text") or row.get("material"))
    ctx["dimensions"] = _s(row.get("dimensions") or row.get("dimensions_raw"))
    ctx["care_instructions"] = _s(row.get("care_instructions"))
    # Other
    ctx["colors"] = _s(row.get("color_options") or row.get("colors") or row.get("selected_color"))
    ctx["price"] = row.get("price_current") or row.get("price") or row.get("price_currency")
    if ctx["price"] is None:
        ctx["price"] = ""
    ctx["bag_structure"] = _s(row.get("bag_structure"))
    ctx["closure_type"] = _s(row.get("closure_type"))
    ctx["handle_type"] = _s(row.get("handle_type"))
    ctx["interior_features"] = _s(row.get("interior_features"))
    ctx["exterior_features"] = _s(row.get("exterior_features"))
    ctx["specs"] = _s(row.get("specs"))
    return ctx


def _build_context_str(ctx: dict) -> str:
    """Build a concise context string for the LLM prompt."""
    parts = []
    if ctx.get("name"):
        parts.append(f"Name: {ctx['name']}")
    if ctx.get("brand"):
        parts.append(f"Brand: {ctx['brand']}")
    if ctx.get("category"):
        parts.append(f"Category: {ctx['category']}")
    if ctx.get("material"):
        parts.append(f"Material: {ctx['material']}")
    if ctx.get("dimensions"):
        parts.append(f"Dimensions: {ctx['dimensions']}")
    if ctx.get("colors"):
        parts.append(f"Colors: {ctx['colors']}")
    if ctx.get("bag_structure"):
        parts.append(f"Structure: {ctx['bag_structure']}")
    if ctx.get("closure_type"):
        parts.append(f"Closure: {ctx['closure_type']}")
    if ctx.get("handle_type"):
        parts.append(f"Handle: {ctx['handle_type']}")
    if ctx.get("interior_features"):
        parts.append(f"Interior: {ctx['interior_features']}")
    if ctx.get("exterior_features"):
        parts.append(f"Exterior: {ctx['exterior_features']}")
    if ctx.get("feature_bullets"):
        parts.append(f"Features: {ctx['feature_bullets'][:400]}")
    if ctx.get("product_details") and not ctx.get("description"):
        parts.append(f"Details: {ctx['product_details'][:400]}")
    if ctx.get("care_instructions"):
        parts.append(f"Care: {ctx['care_instructions'][:150]}")
    if ctx.get("price"):
        parts.append(f"Price: {ctx['price']}")
    return "\n".join(parts)


def _call_azure_openai(prompt: str) -> str:
    """Call Azure OpenAI completion."""
    import openai

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    if not api_key or not endpoint or not deployment:
        raise ValueError(
            "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME"
        )
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint.lstrip("/")
    endpoint = endpoint.rstrip("/")
    client = openai.AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


def _generate_descriptions_batch(products: list[dict], sources: list[str]) -> list[str]:
    """Call LLM to generate descriptions for a batch of products."""
    product_blocks = []
    for i, (row, source) in enumerate(zip(products, sources)):
        ctx = _gather_context(row, source)
        context_str = _build_context_str(ctx)
        product_blocks.append(
            f"--- Product {i} ---\n{context_str}"
        )

    prompt = f"""You are a retail copywriter. Write short, factual product descriptions for bags/handbags.
Each product has context below. Write 1-3 sentences (30-80 words) describing the product for shoppers.
Be specific: mention material, style, key features, and use case. Match the tone of e-commerce product pages.
Do not invent details not present in the context. Do not use marketing fluff.

Products:
{chr(10).join(product_blocks)}

Return valid JSON only, no markdown:
{{"results": [{{"product_index": 0, "description": "..."}}, ...]}}

JSON:"""

    text = _call_azure_openai(prompt)

    # Parse JSON
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON, using raw text")
        return [text[:500] if products else ""]

    results = data.get("results", [])
    out = [""] * len(products)
    for r in results:
        idx = r.get("product_index", -1)
        desc = _s(r.get("description"))
        if 0 <= idx < len(products) and desc:
            out[idx] = desc
    return out


def fill_descriptions_in_rows(
    rows: list[dict],
    *,
    batch_size: int = 8,
    delay_sec: float = 1.5,
    min_desc_len: int = 30,
) -> int:
    """Fill missing descriptions in normalized product rows (in-place).
    Rows should have: name, description, product_details, material_text, category_breadcrumb,
    source (or product_url for source detection). Returns count of descriptions filled.
    """
    need_fill: list[tuple[int, dict, str]] = []
    for i, row in enumerate(rows):
        source = _s(row.get("source")) or _detect_source(row, Path(""))
        desc = _s(row.get("description") or "")
        if len(desc) < min_desc_len:
            need_fill.append((i, row, source or "unknown"))

    if not need_fill:
        return 0

    filled = 0
    for start in range(0, len(need_fill), batch_size):
        if start > 0 and delay_sec > 0:
            time.sleep(delay_sec)
        batch = need_fill[start : start + batch_size]
        products = [r for _, r, _ in batch]
        sources = [s for _, _, s in batch]
        try:
            descriptions = _generate_descriptions_batch(products, sources)
            for (idx, row, _), desc in zip(batch, descriptions):
                if desc:
                    row["description"] = desc
                    filled += 1
        except Exception as e:
            logger.warning("LLM description batch failed: %s", e)
    return filled


def _get_description_column(row: dict, source: str) -> str:
    """Return the column name used for description in this schema."""
    if source == "target" and "description" in row:
        return "description"
    for col in ("description", "product_details"):
        if col in row:
            return col
    return "description"


def process_file(
    path: Path,
    output_dir: Path | None,
    batch_size: int = 8,
    min_desc_len: int = 30,
    delay_sec: float = 1.0,
) -> Path:
    """Process one CSV: fill missing descriptions, write output."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or list(rows[0].keys()) if rows else []

    if "description" not in fieldnames:
        fieldnames = list(fieldnames) + ["description"]

    need_fill: list[tuple[int, dict, str]] = []
    for i, row in enumerate(rows):
        source = _detect_source(row, path)
        desc_col = _get_description_column(row, source)
        desc = _s(row.get(desc_col) or row.get("description") or "")
        if len(desc) < min_desc_len:
            need_fill.append((i, row, source))

    if not need_fill:
        logger.info("%s: All %d rows have descriptions, nothing to fill.", path.name, len(rows))
        out_path = output_dir / path.name if output_dir else path
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return out_path

    logger.info("%s: Filling %d/%d descriptions (source detected from content).", path.name, len(need_fill), len(rows))

    filled = 0
    for start in range(0, len(need_fill), batch_size):
        if start > 0 and delay_sec > 0:
            time.sleep(delay_sec)
        batch = need_fill[start : start + batch_size]
        products = [r for _, r, _ in batch]
        sources = [s for _, _, s in batch]
        try:
            descriptions = _generate_descriptions_batch(products, sources)
            for (idx, row, source), desc in zip(batch, descriptions):
                if desc:
                    rows[idx]["description"] = desc
                    filled += 1
        except Exception as e:
            logger.warning("LLM batch failed: %s", e)

    logger.info("%s: Filled %d descriptions.", path.name, filled)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / path.name
    else:
        out_path = path.parent / (path.stem + "_filled" + path.suffix)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing product descriptions using LLM.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input CSV files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: write _filled suffix alongside input).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Products per LLM batch (default: 8).",
    )
    parser.add_argument(
        "--min-desc-len",
        type=int,
        default=30,
        help="Minimum description length to skip (default: 30).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between LLM batches (default: 1.5) to avoid rate limits.",
    )
    args = parser.parse_args()

    for inp in args.inputs:
        inp = Path(inp)
        if inp.is_dir():
            for f in sorted(inp.glob("*.csv")):
                try:
                    process_file(f, args.output_dir, args.batch_size, args.min_desc_len, args.delay)
                except Exception as e:
                    logger.error("%s: %s", f, e)
        else:
            try:
                process_file(inp, args.output_dir, args.batch_size, args.min_desc_len, args.delay)
            except Exception as e:
                logger.error("%s: %s", inp, e)

    logger.info("Done.")


if __name__ == "__main__":
    main()
