#!/usr/bin/env python3
"""
K-Means clustering for Target + Gap handbags: category & style clusters.

Uses normalized combined dataset by default (products_normalized_with_description.csv).
Builds text embeddings (sentence-transformers or TF-IDF+SVD) + structured numeric
features, then clusters for competitor mapping.

Usage:
    python scripts/run_product_clustering.py
    python scripts/run_product_clustering.py --normalized output/products_normalized_with_description.csv -k 12
    python scripts/run_product_clustering.py --target data/target_*.csv --gap data/gap_*.csv -k 12
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NORMALIZED = PROJECT_ROOT / "output" / "products_normalized_with_description.csv"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "target_handbags_20260222_183113.csv"
DEFAULT_GAP = PROJECT_ROOT / "data" / "gap_products_export_85f1f4a2.csv"
DEFAULT_OUT = PROJECT_ROOT / "output"


def _s(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s in ("", "nan", "None", "{}") else s


def _f(v, default=0.0):
    try:
        x = float(v)
        return x if not np.isnan(x) else default
    except (TypeError, ValueError):
        return default


# --- Promo/spec stripping: prevents clusters from splitting on boilerplate ---
# "Customers also viewed" / "You may also like" blocks (Gap care_instructions scraped junk)
_VIEWED_LIKE_PATTERN = re.compile(
    r"(?:Shipping\s*&\s*returns\s+)?(?:Customers\s+also\s+viewed|You\s+may\s+also\s+like).*",
    re.I | re.S,
)
# Promo phrases (Gap: "extra 10% off with app", sale banners, inline prices)
_PROMO_PATTERNS = [
    re.compile(r"\d+%\s*off\s*\+\s*extra\s+10%\s*in\s+the\s+app", re.I),
    re.compile(r"(?:extra\s+)?\d+%\s*off?\s*in\s+the\s+app", re.I),
    re.compile(r"\bextra\s+\d+%?\s*off\b", re.I),
    re.compile(r"\b\d+%\s*off\b", re.I),
    re.compile(r"\boff\s+with\s+(the\s+)?app\b", re.I),
    re.compile(r"\bwith\s+(the\s+)?app\b", re.I),
    re.compile(r"\b(?:in\s+the\s+)?(?:Gap\s+)?[Aa]pp\b", re.I),
    re.compile(r"\blimited\s+time\b", re.I),
    re.compile(r"\bexcluded\s+from\s+promotions?\b", re.I),
    re.compile(r"\bpromo\s+excluded\b", re.I),
    re.compile(r"\$\d+\.?\d*(?:\s+\$\d+\.?\d*)?", re.I),  # $19.95, $69.95 $27.00
    re.compile(r"\bSold\s*&\s*shipped\s+by\s+[^.]*\.?\s*", re.I),
    re.compile(r"\bReturn\s+by\s+mail\s+only\.?\s*", re.I),
    re.compile(r"\bAdd\s+to\s+Bag\b", re.I),
    re.compile(r"\bGet\s+it\s+before\s+it's\s+gone\b", re.I),
    re.compile(r"\bOnly\s+a\s+few\s+left!?\s*", re.I),
    re.compile(r"\.?\s*For\s+every\s+bag\s+sold[^.]*donates[^.]*\.?", re.I),
]
# Spec token junk (Target: "(h)(w)(d)", measurement template noise)
# Each item: (pattern, replacement) - use " " to remove, r"\1" to keep capture
_SPEC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\([HWDhwd]\)", re.I), " "),
    (re.compile(r"\b\d+\s*Inches?\s*\(\s*[HWD]\s*\)", re.I), " "),
    (re.compile(r"\b(\d+)\s+Inches?\b", re.I), r"\1"),  # collapse "15 Inches" -> "15"
    (re.compile(r"\bFeatures?\s*:\s*", re.I), " "),
    (re.compile(r"\bDetails?\s*:\s*", re.I), " "),
    (re.compile(r'[{}""]', re.I), " "),
]


def clean_text_for_embedding(text: str) -> str:
    """
    Strip promo boilerplate and spec token noise before embedding.
    Prevents clusters from splitting on "extra 10% off app" and "(h)(w) inches".
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text
    t = _VIEWED_LIKE_PATTERN.sub(" ", t)
    for p in _PROMO_PATTERNS:
        t = p.sub(" ", t)
    for pat, repl in _SPEC_PATTERNS:
        t = pat.sub(repl, t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\|\s*\|", "|", t)  # collapse empty segments
    return t.strip()


def clean_gap_dimensions(x: str) -> str:
    """Remove UI boilerplate from Gap dimensions."""
    if not isinstance(x, str):
        return ""
    bad = [
        "Sold & shipped by",
        "Return by mail only",
        "Add to Bag",
        "Get it before it's gone",
        "Only a few left!",
        r"\d+\s*$",
    ]
    for b in bad:
        if b.startswith("\\"):
            x = re.sub(b, "", x)
        else:
            x = x.replace(b, "")
    return " ".join(x.split()).strip()


def flatten_specs(specs_raw: str) -> str:
    """Parse Target specs JSON and flatten to 'key: value' text."""
    if not specs_raw or not isinstance(specs_raw, str) or specs_raw.strip() in ("", "{}"):
        return ""
    s = specs_raw.strip()
    if not s.startswith("{"):
        return s
    try:
        d = json.loads(s.replace("'", '"'))
        if not isinstance(d, dict):
            return s
        return " ".join(f"{k}: {v}" for k, v in d.items() if v)
    except Exception:
        return s


def build_text_target(row: dict) -> str:
    """Target text template for embedding."""
    name_desc = f"{_s(row.get('title'))} {_s(row.get('description'))}"
    inferred = extract_category_from_description(name_desc)
    cat_exp = expand_category_breadcrumb(_s(row.get("breadcrumb")) or _s(row.get("category_breadcrumb")))
    category_block = " ".join(filter(None, [inferred, cat_exp])).strip()
    parts = [
        _s(row.get("brand")),
        _s(row.get("title")),
        category_block,
        _s(row.get("leaf_category")),
        _s(row.get("material")),
        _s(row.get("closure_type")),
        _s(row.get("handle_type")),
        _s(row.get("bag_structure")),
        _s(row.get("interior_features")),
        _s(row.get("exterior_features")),
        _s(row.get("dimensions_raw")),
        flatten_specs(_s(row.get("specs"))),
        _s(row.get("description")),
        _s(row.get("fabric_name")),
    ]
    raw = " | ".join(p for p in parts if p)
    return clean_text_for_embedding(raw)


def build_text_gap(row: dict) -> str:
    """Gap text template for embedding."""
    dims = clean_gap_dimensions(_s(row.get("dimensions")) + " " + _s(row.get("dimensions_section")))
    name_desc = f"{_s(row.get('name'))} {_s(row.get('description'))}"
    inferred = extract_category_from_description(name_desc)
    cat_exp = expand_category_breadcrumb(_s(row.get("category_breadcrumb")))
    category_block = " ".join(filter(None, [inferred, cat_exp])).strip()
    parts = [
        _s(row.get("brand")),
        _s(row.get("name")),
        category_block,
        _s(row.get("description")),
        _s(row.get("product_details")),
        dims,
        _s(row.get("material_text")) or _s(row.get("materials_section")),
        _s(row.get("color_options")),
        _s(row.get("size_options")),
    ]
    raw = " | ".join(p for p in parts if p)
    return clean_text_for_embedding(raw)


# Product-type terms to extract from name/description as inferred category (handbag/bag taxonomy)
_PRODUCT_TYPE_TERMS = frozenset({
    "tote", "totes", "crossbody", "cross-body", "satchel", "satchels",
    "clutch", "clutches", "shoulder", "backpack", "backpacks",
    "duffel", "duffels", "weekender", "weekenders", "messenger",
    "hobo", "wristlet", "wristlets", "pouch", "pouches",
    "bucket", "saddle", "envelope", "structured", "unstructured",
    "charm", "charms", "keychain", "keychains", "organizer", "organisers",
})


def extract_category_from_description(text: str) -> str:
    """
    Extract product-type terms from name + description to infer category.
    Returns space-separated matched terms (e.g. "tote charm keychain") for products
    where the retailer breadcrumb may not include them (e.g. Gap "Bags & Bag Charms").
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    found = []
    for w in words:
        if w in _PRODUCT_TYPE_TERMS and w not in found:
            found.append(w)
    return " ".join(found) if found else ""


def expand_category_breadcrumb(breadcrumb: str) -> str:
    """
    Parse category_breadcrumb into individual category tokens for stronger embedding signal.
    E.g. "Target > Clothing > Handbags > Totes" -> "Clothing Handbags Totes"
    "Women / Bags, Scarves & More > Bags" -> "Women Bags Scarves More Bags"
    """
    if not breadcrumb or not isinstance(breadcrumb, str):
        return ""
    text = breadcrumb.strip()
    if not text:
        return ""
    # Split on common separators: > / , &
    segments = re.split(r"\s*[>/,]\s*|\s+&\s+", text)
    tokens = []
    skip = {"target", "all_products", "shop", "all"}
    for seg in segments:
        s = seg.strip()
        if not s or s.lower() in skip:
            continue
        tokens.append(s)
    return " ".join(tokens) if tokens else ""


def build_text_normalized(row: dict) -> str:
    """
    Unified normalized schema text template for embedding.
    Category first: infer from description (tote, crossbody, charm, etc.), then enhance with
    category_breadcrumb. Description-driven category anchors clustering when breadcrumb is sparse.
    """
    dims = clean_gap_dimensions(_s(row.get("dimensions")) + " " + _s(row.get("dimensions_section")))
    name = _s(row.get("name"))
    desc = _s(row.get("description"))

    # 1) Infer category from name + description (tote, crossbody, charm, clutch, etc.)
    inferred_category = extract_category_from_description(f"{name} {desc}")

    # 2) Expand retailer breadcrumb for additional hierarchy
    breadcrumb_expanded = expand_category_breadcrumb(_s(row.get("category_breadcrumb")))
    leaf = _s(row.get("leaf_category"))

    # Channel A: Category anchor — inferred first (from description), then breadcrumb
    category_block = " ".join(filter(None, [inferred_category, breadcrumb_expanded, leaf])).strip()

    # Channel B: Description + attributes (semantic detail)
    parts = [
        _s(row.get("brand")),
        _s(row.get("name")),
        _s(row.get("description")),
        _s(row.get("product_details")),
        _s(row.get("material_text")) or _s(row.get("materials_section")),
        dims,
        _s(row.get("bag_structure")),
        _s(row.get("interior_features")),
        _s(row.get("exterior_features")),
        _s(row.get("closure_type")),
        _s(row.get("handle_type")),
        _s(row.get("fabric_name")),
        _s(row.get("color_options")),
        _s(row.get("size_options")),
    ]
    detail_block = " | ".join(p for p in parts if p)

    # Combine: category first (reinforced), then detail
    if category_block and detail_block:
        raw = f"{category_block} | {detail_block}"
    else:
        raw = category_block or detail_block

    return clean_text_for_embedding(raw)


def load_normalized(path: Path) -> pd.DataFrame:
    """Load normalized combined CSV (unified schema)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "source" not in df.columns:
        raise ValueError("Normalized CSV must have 'source' column")
    df["product_id"] = df["product_id"].astype(str)
    df["text"] = df.apply(build_text_normalized, axis=1)

    # Ensure required columns for clustering
    if "image_count" not in df.columns or df["image_count"].isna().all():
        img_raw = df.get("images_list", pd.Series([""] * len(df))).fillna("")
        df["image_count"] = img_raw.str.split(r"\s*\|\s*").apply(
            lambda x: len([u for u in x if u.strip() and "http" in str(u)])
        ).clip(upper=50)
    else:
        df["image_count"] = df["image_count"].fillna(0).astype(int).clip(upper=50)

    return df


def load_target(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source"] = "target"
    df["product_id"] = df["tcin"].astype(str)
    df["text"] = df.apply(build_text_target, axis=1)
    return df


def load_gap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source"] = "gap"
    df["product_id"] = df["product_id"].astype(str)
    df["text"] = df.apply(build_text_gap, axis=1)
    return df


def make_numeric_target(df: pd.DataFrame) -> np.ndarray:
    """Structured numeric features for Target."""
    price = df["price"].replace(0, np.nan).fillna(df["price"].median())
    review = np.log1p(df["review_count"].fillna(0).astype(float))
    rating = df["rating"].replace(0, np.nan).fillna(df["rating"].median())
    sale = df["is_on_sale"].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0)
    img_raw = df["images"].fillna("")
    img_count = img_raw.str.split(r"\s*\|\s*").apply(lambda x: len([u for u in x if u.strip() and "http" in str(u)]))
    img_count = img_count.clip(upper=50)
    return np.column_stack([price, sale, rating, review, img_count])


def make_numeric_gap(df: pd.DataFrame) -> np.ndarray:
    """Structured numeric features for Gap."""
    price = df["price_current"].fillna(0).astype(float).replace(0, np.nan)
    if price.isna().all():
        price = pd.Series([0.0] * len(df))
    else:
        price = price.fillna(price.median())
    review = np.log1p(df["review_count"].fillna(0).astype(float))
    rating = df["average_rating"].fillna(0).astype(float).replace(0, np.nan)
    if rating.isna().all():
        rating = pd.Series([0.0] * len(df))
    else:
        rating = rating.fillna(rating.median())
    sale = df["is_sale"].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0)
    img_count = df["image_count"].fillna(0).astype(int).clip(upper=50)
    return np.column_stack([price, sale, rating, review, img_count])


# Generic words to exclude when building cluster labels
_LABEL_STOPWORDS = frozenset(
    {"bag", "bags", "your", "closure", "zip", "pocket", "accessories", "women",
     "size", "one", "more", "guide", "cotton", "strap", "handle", "compartments"}
)


def _cluster_label(top_terms: list[str], median_price: float, n_target: int, n_gap: int) -> str:
    """Generate a human-readable label for a cluster from top terms and stats."""
    terms = {t.lower() for t in top_terms}
    # Brand / product-type keywords (order matters for specificity)
    if "charm" in terms or "keychain" in terms or "shiraleah" in terms:
        return "Charms & Keychains"
    if "kipling" in terms:
        return "Kipling Nylon Bags"
    if "vera" in terms:
        return "Vera Bradley Crossbody"
    if "champion" in terms:
        return "Champion Crossbody / Mini"
    if "state" in terms or "donates" in terms:
        return "STATE Premium (Charity)"
    if "logo" in terms and "tote" in terms:
        return "Logo Totes"
    if "leather" in terms and median_price > 70:
        return "Premium Leather Handbags"
    if "leather" in terms:
        return "Leather Bags"
    if "nylon" in terms and median_price > 80:
        return "Premium Nylon / Travel"
    if "weekender" in terms or "duffel" in terms:
        return "Weekenders & Duffels"
    if "tote" in terms:
        return "Totes"
    if "crossbody" in terms or "mini" in terms:
        return "Crossbody / Mini Bags"
    if "backpack" in terms:
        return "Backpacks"
    if "kids" in terms:
        return "Kids Bags & Charms"
    if "shoulder" in terms and "compartments" in terms:
        return "Shoulder Bags w/ Compartments"
    if "shoulder" in terms:
        return "Shoulder Bags"
    # Fallback: use first 2 distinctive terms
    distinctive = [t for t in top_terms[:5] if t.lower() not in _LABEL_STOPWORDS]
    if len(distinctive) >= 2:
        return " ".join(distinctive[:2]).title()
    if distinctive:
        return (distinctive[0].title() + " Bags")
    # Price-tier fallback
    if median_price < 25:
        return "Budget Bags"
    if median_price > 80:
        return "Premium Bags"
    return "Mid-Range Bags"


def _plot_clustering(
    combined: pd.DataFrame,
    cluster_summary: list[dict],
    X: np.ndarray,
    output_dir: Path,
    ts: str,
) -> list[Path]:
    """Generate clustering visualizations. Returns list of saved PNG paths."""
    saved: list[Path] = []
    n_clusters = len(cluster_summary)

    # Color palette (tab20 has 20 colors, cycle if needed)
    cmap = plt.cm.tab20
    cluster_colors = [cmap(i % 20) for i in range(n_clusters)]

    labels = [s.get("label", f"C{s['cluster']}") for s in cluster_summary]
    # Shorten labels for chart axis (max 18 chars)
    short_labels = [lbl[:18] + "…" if len(lbl) > 18 else lbl for lbl in labels]

    # 1) Cluster distribution stacked bar (Target vs Gap)
    fig, ax = plt.subplots(figsize=(10, 5))
    target_counts = [s["target"] for s in cluster_summary]
    gap_counts = [s["gap"] for s in cluster_summary]
    x_pos = np.arange(n_clusters)
    ax.bar(x_pos, target_counts, label="Target", color="#CC0000", alpha=0.9)
    ax.bar(x_pos, gap_counts, bottom=target_counts, label="Gap", color="#0066CC", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Product count")
    ax.set_title("Cluster distribution by retailer (Target vs Gap)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p1 = output_dir / f"cluster_distribution_{ts}.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p1)

    # 2) Median price by cluster
    fig, ax = plt.subplots(figsize=(10, 5))
    med_prices = [s["median_price"] for s in cluster_summary]
    bars = ax.bar(x_pos, med_prices, color=cluster_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Median price ($)")
    ax.set_title("Median price by cluster")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p2 = output_dir / f"cluster_price_{ts}.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p2)

    # 3) 2D scatter (PCA projection, colored by cluster)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(n_clusters):
        mask = combined["cluster"] == c
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            c=[cluster_colors[c]],
            label=short_labels[c],
            alpha=0.6,
            s=25,
            edgecolors="white",
            linewidth=0.3,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Clusters in 2D (PCA projection)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    p3 = output_dir / f"cluster_scatter_{ts}.png"
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p3)

    # 4) Retailer share % by cluster (grouped bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    tgt_pct = [100 * s["target"] / s["count"] if s["count"] else 0 for s in cluster_summary]
    gap_pct = [100 * s["gap"] / s["count"] if s["count"] else 0 for s in cluster_summary]
    width = 0.35
    ax.bar(x_pos - width / 2, tgt_pct, width, label="Target %", color="#CC0000", alpha=0.9)
    ax.bar(x_pos + width / 2, gap_pct, width, label="Gap %", color="#0066CC", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Share (%)")
    ax.set_title("Retailer share by cluster (%)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p4 = output_dir / f"cluster_retailer_share_{ts}.png"
    fig.savefig(p4, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p4)

    return saved


def run_clustering(
    normalized_path: Path | None = None,
    target_path: Path | None = None,
    gap_path: Path | None = None,
    k: int = 12,
    output_dir: Path = DEFAULT_OUT,
    model_name: str = "all-MiniLM-L6-v2",
    use_tfidf: bool = False,
) -> None:
    # Load: normalized (default) or raw target+gap
    if normalized_path is not None and normalized_path.exists():
        logger.info("Loading normalized dataset %s ...", normalized_path)
        combined = load_normalized(normalized_path)
        n_target = int((combined["source"] == "target").sum())
        n_gap = int((combined["source"] == "gap").sum())
        logger.info("  %d rows (%d target, %d gap)", len(combined), n_target, n_gap)
        # Select/ensure columns for downstream
        cols = ["source", "product_id", "text", "price_current", "average_rating", "review_count", "is_sale", "name", "image_count"]
        missing = [c for c in cols if c not in combined.columns]
        if missing:
            raise ValueError(f"Normalized CSV missing columns: {missing}")
        combined = combined[cols].copy()
        combined["price_current"] = combined["price_current"].astype(float)
    else:
        if target_path is None or gap_path is None:
            raise ValueError("Provide --normalized path or both --target and --gap")
        logger.info("Loading Target %s ...", target_path)
        target = load_target(target_path)
        logger.info("  %d rows", len(target))
        logger.info("Loading Gap %s ...", gap_path)
        gap = load_gap(gap_path)
        logger.info("  %d rows", len(gap))

        target_union = target[
            ["source", "product_id", "text", "price", "rating", "review_count", "images", "is_on_sale", "title"]
        ].copy()
        target_union = target_union.rename(
            columns={
                "price": "price_current",
                "rating": "average_rating",
                "is_on_sale": "is_sale",
                "title": "name",
            }
        )
        target_union["price_current"] = target["price"].astype(float)
        img_split = target_union["images"].fillna("").str.split(r"\s*\|\s*")
        target_union["image_count"] = img_split.apply(
            lambda x: len([u for u in x if u.strip() and "http" in str(u)])
        ).clip(upper=50)

        gap_union = gap[
            ["source", "product_id", "text", "price_current", "average_rating", "review_count", "images_list", "is_sale", "name", "image_count"]
        ].copy()
        gap_union["price_current"] = gap_union["price_current"].astype(float)
        gap_union["image_count"] = gap["image_count"].fillna(0).astype(int).clip(upper=50)

        combined = pd.concat([target_union, gap_union], ignore_index=True)
        logger.info("Combined: %d rows (%d target, %d gap)", len(combined), len(target), len(gap))

    # Embeddings
    texts = combined["text"].fillna("").tolist()
    texts = [t if t.strip() else "(no description)" for t in texts]

    if use_tfidf:
        logger.info("Using TF-IDF + SVD (--tfidf)")

    if not use_tfidf:
        try:
            logger.info("Loading SentenceTransformer %s ...", model_name)
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            logger.info("Encoding %d texts ...", len(texts))
            E = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        except Exception as ex:
            logger.warning("SentenceTransformer failed (%s), falling back to TF-IDF+SVD", ex)
            use_tfidf = True

    if use_tfidf:
        logger.info("Building TF-IDF vectors ...")
        tfidf = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.95, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=128, random_state=42)
        E = svd.fit_transform(X_tfidf)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1
        E = E / norms

    # Numeric features (unified)
    price = combined["price_current"].fillna(0).replace(0, np.nan)
    price = price.fillna(price.median()).values
    review = np.log1p(combined["review_count"].fillna(0).astype(float).values)
    rating = combined["average_rating"].fillna(0).astype(float).replace(0, np.nan).values
    rating = np.nan_to_num(rating, nan=np.nanmedian(rating))
    sale = combined["is_sale"].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0).values
    img_count = combined["image_count"].fillna(0).astype(int).clip(upper=50).values

    num = np.column_stack([price, sale, rating, review, img_count])
    num_scaled = StandardScaler().fit_transform(num)

    X = np.hstack([E, num_scaled])
    logger.info("Feature matrix: %s", X.shape)

    # K-Means
    logger.info("K-Means with k=%d ...", k)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    combined["cluster"] = km.fit_predict(X)

    sil = silhouette_score(X, combined["cluster"])
    db = davies_bouldin_score(X, combined["cluster"])
    logger.info("Silhouette: %.4f (higher=better)", sil)
    logger.info("Davies-Bouldin: %.4f (lower=better)", db)

    # Cluster summary
    cluster_summary = []
    for c in range(k):
        mask = combined["cluster"] == c
        sub = combined[mask]
        n_target = (sub["source"] == "target").sum()
        n_gap = (sub["source"] == "gap").sum()
        med_price = sub["price_current"].median()
        sale_pct = 100 * (sub["is_sale"].map(lambda x: str(x).lower() in ("true", "1", "yes")).sum() / len(sub))

        sub_tgt = sub[sub["source"] == "target"]
        sub_gap = sub[sub["source"] == "gap"]
        med_price_target = float(sub_tgt["price_current"].median()) if len(sub_tgt) else None
        med_price_gap = float(sub_gap["price_current"].median()) if len(sub_gap) else None
        sale_target = (
            100 * (sub_tgt["is_sale"].map(lambda x: str(x).lower() in ("true", "1", "yes")).sum() / len(sub_tgt))
            if len(sub_tgt) else None
        )
        sale_gap = (
            100 * (sub_gap["is_sale"].map(lambda x: str(x).lower() in ("true", "1", "yes")).sum() / len(sub_gap))
            if len(sub_gap) else None
        )

        # Top terms from text (simple word freq)
        words = " ".join(sub["text"].fillna("")).lower().split()
        words = [w for w in words if len(w) > 2 and w not in ("the", "and", "for", "with", "from", "this", "that")]
        top_terms = [w for w, _ in Counter(words).most_common(15)]

        label = _cluster_label(top_terms, float(med_price), int(n_target), int(n_gap))
        cluster_summary.append(
            {
                "cluster": int(c),
                "label": label,
                "count": int(len(sub)),
                "target": int(n_target),
                "gap": int(n_gap),
                "median_price": float(med_price),
                "sale_pct": round(float(sale_pct), 1),
                "median_price_target": float(med_price_target) if med_price_target is not None else None,
                "median_price_gap": float(med_price_gap) if med_price_gap is not None else None,
                "sale_pct_target": round(float(sale_target), 1) if sale_target is not None else None,
                "sale_pct_gap": round(float(sale_gap), 1) if sale_gap is not None else None,
                "top_terms": top_terms[:10],
            }
        )

    # Ensure unique labels (append cluster id if duplicate)
    seen: dict[str, int] = {}
    for s in cluster_summary:
        lbl = s["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
    for s in cluster_summary:
        if seen.get(s["label"], 0) > 1:
            s["label"] = f"{s['label']} (C{s['cluster']})"

    # Write outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cluster_to_label = {s["cluster"]: s["label"] for s in cluster_summary}
    combined["cluster_label"] = combined["cluster"].map(cluster_to_label)

    out_csv = output_dir / f"clustered_products_{ts}.csv"
    combined.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("  → %s", out_csv)

    out_summary = output_dir / f"cluster_summary_{ts}.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "k": int(k),
                "silhouette": float(sil),
                "davies_bouldin": float(db),
                "n_total": int(len(combined)),
                "n_target": int((combined["source"] == "target").sum()),
                "n_gap": int((combined["source"] == "gap").sum()),
                "clusters": cluster_summary,
            },
            f,
            indent=2,
        )
    logger.info("  → %s", out_summary)

    # Report
    out_report = output_dir / f"cluster_report_{ts}.md"
    lines = [
        "# Product Clustering Report",
        "",
        f"**Date:** {ts}",
        f"**K:** {k} | **Silhouette:** {sil:.4f} | **Davies-Bouldin:** {db:.4f}",
        "",
        "## Cluster Summary",
        "",
        "| Label | Count | Target | Gap | Med Price | Sale % | Top Terms |",
        "|-------|-------|--------|-----|-----------|--------|-----------|",
    ]
    for s in cluster_summary:
        terms = ", ".join(s["top_terms"][:6])
        lines.append(
            f"| {s['label']} | {s['count']} | {s['target']} | {s['gap']} | ${s['median_price']:.1f} | {s['sale_pct']}% | {terms} |"
        )
    lines.extend(["", "## Retailer share by cluster", ""])
    for s in cluster_summary:
        tgt_pct = 100 * s["target"] / s["count"] if s["count"] else 0
        gap_pct = 100 * s["gap"] / s["count"] if s["count"] else 0
        med_t = f"${s['median_price_target']:.1f}" if s.get("median_price_target") is not None else "—"
        med_g = f"${s['median_price_gap']:.1f}" if s.get("median_price_gap") is not None else "—"
        sale_t = f"{s['sale_pct_target']}%" if s.get("sale_pct_target") is not None else "—"
        sale_g = f"{s['sale_pct_gap']}%" if s.get("sale_pct_gap") is not None else "—"
        lines.append(
            f"- **{s['label']}:** Target {tgt_pct:.0f}% (med {med_t} / sale {sale_t}) | Gap {gap_pct:.0f}% (med {med_g} / sale {sale_g})"
        )

    if HAS_MATPLOTLIB:
        chart_files = _plot_clustering(
            combined=combined,
            cluster_summary=cluster_summary,
            X=X,
            output_dir=output_dir,
            ts=ts,
        )
        for p in chart_files:
            logger.info("  → %s", p)
        lines.extend(["", "## Visualizations", ""])
        for p in chart_files:
            lines.append(f"![{p.name}]({p.name})")
            lines.append("")

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("  → %s", out_report)

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means clustering for Target + Gap handbags (category/style clusters)."
    )
    parser.add_argument(
        "--normalized",
        "-n",
        type=Path,
        default=DEFAULT_NORMALIZED,
        help="Normalized combined CSV (default: output/products_normalized_with_description.csv)",
    )
    parser.add_argument("--target", type=Path, default=None, help="Target CSV (use with --gap, overrides --normalized)")
    parser.add_argument("--gap", type=Path, default=None, help="Gap CSV (use with --target)")
    parser.add_argument("-k", type=int, default=12, help="Number of clusters (default: 12)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--tfidf",
        action="store_true",
        help="Use TF-IDF + SVD instead of sentence-transformers (no torch needed)",
    )
    args = parser.parse_args()

    use_raw = args.target is not None and args.gap is not None
    normalized_path = None if use_raw else args.normalized
    target_path = args.target if use_raw else DEFAULT_TARGET
    gap_path = args.gap if use_raw else DEFAULT_GAP

    run_clustering(
        normalized_path=normalized_path,
        target_path=target_path,
        gap_path=gap_path,
        k=args.k,
        output_dir=args.output_dir,
        model_name=args.model,
        use_tfidf=args.tfidf,
    )


if __name__ == "__main__":
    main()
