#!/usr/bin/env python3
"""
LLM-based mission tagger using Azure OpenAI (gpt-5-mini).

Assigns multi-label mission tags from the 20-mission taxonomy to products.
Uses product name, brand, category, description, material, price.

Usage:
    from llm_mission_tagger import tag_missions_batch, MISSION_TAXONOMY
    tags = tag_missions_batch([product_row1, product_row2])
"""
from __future__ import annotations

import json
import math
import logging
import os
import time
from typing import Any

try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 20-Mission Taxonomy (ID, label, signals for LLM context)
MISSION_TAXONOMY = [
    ("occasion_glam", "Party, evening, statement", "embellished, metallic, clutch, satin"),
    ("everyday_crossbody", "Hands-free daily", "adjustable strap, medium size, zip top"),
    ("work_tote", "Professional + capacity", "laptop sleeve, structured, compartments"),
    ("weekend_bucket", "Casual soft structure", "bucket shape, relaxed silhouette"),
    ("travel_utility", "Security + packability", "zip, nylon, anti-theft, multi-pocket"),
    ("micro_mini_impulse", "Trend small bags", "mini, novelty, bright colors"),
    ("slg_small_leathers", "Wallets, cardholders", "wallet, cardholder, key case"),
    ("gym_active", "Athleisure", "nylon, duffel, water-resistant"),
    ("diaper_parent", "Parent utility", "stroller straps, bottle holders"),
    ("campus_student", "School/college", "backpack, laptop fit"),
    ("luxury_signal", "Status", "logo, premium leather"),
    ("minimal_modern", "Clean staple", "neutral, low hardware"),
    ("statement_fashion", "Fashion-forward", "bold shape, novelty"),
    ("value_mass", "Price-led", "synthetic, low price"),
    ("eco_conscious", "Sustainability", "recycled, vegan leather"),
    ("gifting_ready", "Occasion gifting", "compact, decorative"),
    ("convertible_multiway", "Functional flexibility", "removable strap, 2-way"),
    ("security_priority", "Safety", "RFID, anti-slash"),
    ("oversized_carryall", "Max capacity", "weekender, large tote"),
    ("festival_outdoor", "Event + outdoor", "crossbody small, durable"),
]

MISSION_IDS = [m[0] for m in MISSION_TAXONOMY]


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
    s = str(v).strip()
    return "" if s in ("", "nan", "None", "{}") else s


def _extract_product_for_prompt(row: dict) -> dict:
    """Extract fields for LLM from product row (ChromaDB metadata or CSV)."""
    return {
        "product_id": _s(row.get("product_id") or row.get("id") or row.get("tcin")),
        "name": _s(row.get("name") or row.get("title") or row.get("product_name")),
        "brand": _s(row.get("brand")),
        "category": _s(row.get("category_breadcrumb") or row.get("leaf_category")),
        "description": _s(row.get("description") or row.get("document") or ""),
        "material": _s(row.get("material_text") or row.get("material")),
        "price": row.get("price_current") or row.get("price") or 0,
    }


def _build_prompt(products: list[dict]) -> str:
    """Build a single prompt for batch of products."""
    mission_block = "\n".join(
        f"- {mid}: {label} (signals: {signals})"
        for mid, label, signals in MISSION_TAXONOMY
    )
    product_block = "\n\n".join(
        f"--- Product {i+1} (id={p['product_id']}) ---\n"
        f"Name: {p['name']}\nBrand: {p['brand']}\nCategory: {p['category']}\n"
        f"Description: {p['description']}\nMaterial: {p['material']}\nPrice: ${p['price']}"
        for i, p in enumerate(products)
    )
    return f"""You are a retail product taxonomy expert. Assign mission tags to handbag/bag products.

MISSIONS (choose 1-4 per product, only from this list):
{mission_block}

RULES:
- Each product can have multiple mission tags (multi-label).
- Use only the mission IDs listed above.
- Consider: name, brand, category, description, material, price.
- Value/low-price products: value_mass.
- Recycled/vegan: eco_conscious.
- Work laptop bags: work_tote.
- Weekender/travel: travel_utility or oversized_carryall.
- Clutches/evening: occasion_glam.
- Mini/trendy: micro_mini_impulse.

Return valid JSON only, no markdown:
{{"results": [{{"product_index": 0, "mission_tags": ["work_tote", "value_mass"]}}, ...]}}

Products (index from 0):
{product_block}

JSON:"""


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
    # Ensure endpoint has protocol; SDK requires https
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
        max_completion_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_llm_response(text: str, n_products: int) -> list[list[str]]:
    """Parse LLM JSON response into list of mission tag lists per product."""
    import re
    result = [[] for _ in range(n_products)]
    t = (text or "").strip()
    if not t:
        logger.warning("LLM returned empty response.")
        return result

    # Remove markdown code blocks
    if "```" in t:
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", t):
            if "results" in block:
                t = block.strip()
                break

    # Try to extract JSON object if wrapped in prose
    if not t.startswith("{"):
        match = re.search(r'\{[\s\S]*"results"[\s\S]*\}', t)
        if match:
            t = match.group(0)

    try:
        data = json.loads(t)
        for item in data.get("results", []):
            idx = item.get("product_index", -1)
            tags = item.get("mission_tags", [])
            if 0 <= idx < n_products and isinstance(tags, list):
                result[idx] = [str(x) for x in tags if str(x) in MISSION_IDS]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("LLM response parse error: %s. Response preview: %s", e, repr(t[:300]) if t else "(empty)")
    return result


def tag_missions_batch(
    products: list[dict],
    batch_size: int = 8,
    batch_delay_sec: float | None = None,
) -> list[list[str]]:
    """
    Tag a batch of products with mission tags via Azure OpenAI (gpt-5-mini).
    Retries indefinitely with exponential backoff (capped at 60s) on failure.
    Paces requests to stay under Azure rate limits (100 req/min).

    Args:
        products: List of product dicts (ChromaDB metadata or CSV rows)
        batch_size: Products per API call (default: 8)
        batch_delay_sec: Seconds between batches (default: from AZURE_OPENAI_BATCH_DELAY_SEC or 0.7)

    Returns:
        List of mission tag lists, one per product
    """
    delay = batch_delay_sec
    if delay is None:
        try:
            delay = float(os.environ.get("AZURE_OPENAI_BATCH_DELAY_SEC", "0.7"))
        except (TypeError, ValueError):
            delay = 0.7
    extracted = [_extract_product_for_prompt(p) for p in products]
    all_tags: list[list[str]] = [[] for _ in products]

    for start in range(0, len(extracted), batch_size):
        batch = extracted[start : start + batch_size]
        prompt = _build_prompt(batch)
        attempt = 0
        while True:
            try:
                resp = _call_azure_openai(prompt)
                batch_tags = _parse_llm_response(resp, len(batch))
                for i, tags in enumerate(batch_tags):
                    all_tags[start + i] = tags
                logger.info("Tagged products %d-%d", start + 1, start + len(batch))
                break
            except Exception as e:
                attempt += 1
                wait = min(2 ** attempt, 60)
                logger.warning("Attempt %d failed: %s. Retrying in %ds...", attempt, e, wait)
                time.sleep(wait)
        time.sleep(delay)
    return all_tags


def tag_missions_single(product: dict) -> list[str]:
    """Tag a single product. Convenience wrapper."""
    return tag_missions_batch([product], batch_size=1)[0]


if __name__ == "__main__":
    import argparse
    import csv
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tag_missions_and_cluster import (
        load_from_chroma,
        load_from_csv,
        update_chroma_metadata,
        _s,
    )

    parser = argparse.ArgumentParser(description="Tag products with mission tags via LLM.")
    parser.add_argument(
        "--collection",
        default=__import__("os").environ.get("CHROMA_COLLECTION", "target_handbags"),
        help="ChromaDB collection (default: target_handbags).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input CSV (overrides ChromaDB).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "products_mission_tagged.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Products per LLM request (default: 8).",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds between batches (default: from AZURE_OPENAI_BATCH_DELAY_SEC or 0.7).",
    )
    parser.add_argument(
        "--force-retag",
        "--full-refresh",
        action="store_true",
        dest="force_retag",
        help="Re-tag all products even if mission_tags exist in DB.",
    )
    parser.add_argument(
        "--no-update-db",
        action="store_true",
        help="Do not persist mission_tags to ChromaDB (default: persist when loading from DB).",
    )
    parser.add_argument("--chroma-api-key", default=__import__("os").environ.get("CHROMA_API_KEY"))
    parser.add_argument("--chroma-tenant", default=__import__("os").environ.get("CHROMA_TENANT"))
    parser.add_argument("--chroma-database", default=__import__("os").environ.get("CHROMA_DATABASE"))
    parser.add_argument("--chroma-url", default=__import__("os").environ.get("CHROMA_HTTP_URL"))
    parser.add_argument("--chroma-user", default=__import__("os").environ.get("CHROMA_HTTP_USER"))
    parser.add_argument("--chroma-password", default=__import__("os").environ.get("CHROMA_HTTP_PASSWORD"))
    args = parser.parse_args()

    if args.input is not None:
        logger.info("Loading from CSV: %s", args.input)
        rows = load_from_csv(args.input)
        chroma_ids = []
    else:
        logger.info("Loading from ChromaDB collection '%s' (whole dataset)...", args.collection)
        rows, _, chroma_ids = load_from_chroma(
            collection_name=args.collection,
            api_key=args.chroma_api_key,
            tenant=args.chroma_tenant,
            database=args.chroma_database,
            chroma_url=args.chroma_url,
            chroma_user=args.chroma_user,
            chroma_password=args.chroma_password,
        )
        logger.info("Loaded %d products from ChromaDB.", len(rows))

    if not rows:
        logger.error("No products loaded.")
        sys.exit(1)

    need_tagging = [(i, r) for i, r in enumerate(rows) if args.force_retag or not _s(r.get("mission_tags"))]
    if not need_tagging:
        logger.info("All products already have mission_tags; use --force-retag to re-tag.")
    else:
        need_indices = [x[0] for x in need_tagging]
        need_rows = [rows[i] for i in need_indices]
        logger.info("Tagging %d products via LLM (batch_size=%d)...", len(need_rows), args.batch_size)
        tag_lists = tag_missions_batch(
            need_rows, batch_size=args.batch_size, batch_delay_sec=args.batch_delay
        )
        for j, i in enumerate(need_indices):
            rows[i]["mission_tags"] = "|".join(tag_lists[j]) if tag_lists[j] else ""
        still_empty = [i for i in need_indices if not _s(rows[i].get("mission_tags"))]
        if still_empty:
            logger.info("Retrying %d products with empty mission_tags...", len(still_empty))
            retry_tags = tag_missions_batch(
                [rows[i] for i in still_empty],
                batch_size=min(5, len(still_empty)),
                batch_delay_sec=args.batch_delay,
            )
            for j, i in enumerate(still_empty):
                rows[i]["mission_tags"] = "|".join(retry_tags[j]) if retry_tags[j] else ""

    if not args.no_update_db and chroma_ids:
        update_chroma_metadata(
            collection_name=args.collection,
            chroma_ids=chroma_ids,
            rows=rows,
            api_key=args.chroma_api_key,
            tenant=args.chroma_tenant,
            database=args.chroma_database,
            chroma_url=args.chroma_url,
            chroma_user=args.chroma_user,
            chroma_password=args.chroma_password,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted(set().union(*(r.keys() for r in rows)))
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else str(v)) for k, v in r.items()})
    logger.info("Wrote %s", args.output)
