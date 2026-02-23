#!/usr/bin/env python3
"""
LLM-based mission tagger using Azure OpenAI (Llama-4-Maverick).

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
        "description": (_s(row.get("description") or row.get("document") or "")[:800]),
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
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not api_key or not endpoint or not deployment:
        raise ValueError(
            "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME"
        )
    client = openai.AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-15-preview",
        azure_endpoint=endpoint,
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_llm_response(text: str, n_products: int) -> list[list[str]]:
    """Parse LLM JSON response into list of mission tag lists per product."""
    result = [[] for _ in range(n_products)]
    try:
        # Remove markdown code blocks if present
        t = text.strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            if t.startswith("json"):
                t = t[4:]
        data = json.loads(t)
        for item in data.get("results", []):
            idx = item.get("product_index", -1)
            tags = item.get("mission_tags", [])
            if 0 <= idx < n_products and isinstance(tags, list):
                result[idx] = [str(t) for t in tags if str(t) in MISSION_IDS]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("LLM response parse error: %s", e)
    return result


def tag_missions_batch(
    products: list[dict],
    batch_size: int = 5,
    retries: int = 2,
) -> list[list[str]]:
    """
    Tag a batch of products with mission tags via Azure OpenAI.

    Args:
        products: List of product dicts (ChromaDB metadata or CSV rows)
        batch_size: Products per API call
        retries: Retries on failure

    Returns:
        List of mission tag lists, one per product
    """
    extracted = [_extract_product_for_prompt(p) for p in products]
    all_tags: list[list[str]] = [[] for _ in products]

    for start in range(0, len(extracted), batch_size):
        batch = extracted[start : start + batch_size]
        prompt = _build_prompt(batch)
        for attempt in range(retries + 1):
            try:
                resp = _call_azure_openai(prompt)
                batch_tags = _parse_llm_response(resp, len(batch))
                for i, tags in enumerate(batch_tags):
                    all_tags[start + i] = tags
                logger.info("Tagged products %d-%d", start + 1, start + len(batch))
                break
            except Exception as e:
                logger.warning("Attempt %d failed: %s", attempt + 1, e)
                if attempt < retries:
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.error("Skipping batch %d-%d after retries", start + 1, start + len(batch))
        time.sleep(0.5)  # Rate limit cushion
    return all_tags


def tag_missions_single(product: dict) -> list[str]:
    """Tag a single product. Convenience wrapper."""
    return tag_missions_batch([product], batch_size=1)[0]
