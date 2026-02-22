#!/usr/bin/env python3
"""
Run gap analysis on Target handbags data: style clustering (K-Means/HDBSCAN),
PCA/UMAP, price segmentation, rating quadrants, brand dominance, color diversity,
title-keyword co-occurrence, and six practical gap analyses. Outputs enriched
CSV/JSON and a markdown report.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from analysis.utils import build_style_text, load_products

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_products_from_chroma(
    collection_name: str = "target_handbags",
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
) -> list[dict]:
    """Load all products from Chroma collection. Returns list of dicts (metadata + id, document)."""
    import os
    import chromadb

    api_key = api_key or os.environ.get("CHROMA_API_KEY")
    if not api_key:
        raise ValueError("CHROMA_API_KEY required for --from-chroma. Set in .env or pass --api-key.")

    kwargs: dict = {"api_key": api_key}
    if tenant or os.environ.get("CHROMA_TENANT"):
        kwargs["tenant"] = tenant or os.environ.get("CHROMA_TENANT")
    if database or os.environ.get("CHROMA_DATABASE"):
        kwargs["database"] = database or os.environ.get("CHROMA_DATABASE")

    client = chromadb.CloudClient(**kwargs)
    collection = client.get_collection(name=collection_name)

    result = collection.get(include=["metadatas", "documents"])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or [""] * len(ids)

    products = []
    for i, pid in enumerate(ids):
        meta = (metadatas[i] if i < len(metadatas) else {}) or {}
        doc = documents[i] if i < len(documents) else ""
        p = dict(meta)
        p["product_id"] = p.get("product_id") or pid
        if doc and not p.get("name") and not p.get("title"):
            p["description"] = (p.get("description") or "") + (" " + doc[:200] if doc else "")
        products.append(p)
    return products
logger = logging.getLogger(__name__)

# Constants
MAX_DOCUMENT_CHARS = 500
PRICE_BANDS = [(0, 20), (20, 40), (40, 70), (70, 120), (120, float("inf"))]
PRICE_BAND_LABELS = ["0-20", "20-40", "40-70", "70-120", "120+"]
# Coverage-gap bands (includes $60–90 mid-tier sweet spot)
COVERAGE_PRICE_BANDS = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 120), (120, 9999)]
COVERAGE_BAND_LABELS = ["0-20", "20-40", "40-60", "60-90", "90-120", "120+"]
RATING_HIGH_THRESHOLD = 4.0
REVIEW_COUNT_HIGH_DEFAULT = 20  # or median-based
TITLE_KEYWORDS = [
    "tote", "clutch", "backpack", "crossbody", "convertible", "quilted",
    "faux leather", "laptop", "organizer", "mini", "drawstring",
]
COLOR_FAMILY_MAP = {
    "black": "black",
    "white": "neutral", "cream": "neutral", "gray": "neutral", "grey": "neutral",
    "beige": "neutral", "tan": "neutral", "ivory": "neutral", "nude": "neutral",
    "pink": "pastel", "blue": "pastel", "lavender": "pastel", "mint": "pastel",
    "yellow": "pastel", "peach": "pastel", "rose": "pastel", "misty rose": "pastel",
    "red": "bold", "navy": "bold", "olive": "bold", "burgundy": "bold",
    "brown": "neutral", "khaki": "neutral", "gold": "bold", "silver": "neutral",
}


def save_umap_scatter(
    df: pd.DataFrame,
    out_path: Path,
    color_by: str = "cluster_kmeans",
    title: str | None = None,
) -> None:
    """Save a simple UMAP scatter plot (PNG) for quick visual inspection."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "UMAP plotting requires matplotlib. Install with: pip install matplotlib"
        ) from e

    needed = {"umap_x", "umap_y", color_by}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot plot UMAP; missing columns: {missing}")

    plot_df = df[["umap_x", "umap_y", color_by]].dropna(subset=["umap_x", "umap_y"]).copy()
    if plot_df.empty:
        raise ValueError("Cannot plot UMAP; no non-null UMAP coordinates.")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    series = plot_df[color_by]
    if pd.api.types.is_numeric_dtype(series):
        sc = ax.scatter(plot_df["umap_x"], plot_df["umap_y"], c=series, s=10, alpha=0.75, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=color_by)
    else:
        cats = series.astype("category")
        codes = cats.cat.codes
        sc = ax.scatter(plot_df["umap_x"], plot_df["umap_y"], c=codes, s=10, alpha=0.75, cmap="tab20")
        # Legend only if the number of categories is reasonable
        cat_names = list(cats.cat.categories)
        if len(cat_names) <= 20:
            handles = []
            for idx, name in enumerate(cat_names):
                handles.append(
                    plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, label=str(name),
                               markerfacecolor=sc.cmap(idx / max(1, len(cat_names) - 1)), markeredgecolor="none")
                )
            ax.legend(handles=handles, title=color_by, loc="best", fontsize=8, title_fontsize=9)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"UMAP scatter (colored by {color_by})")
    ax.grid(False)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _coerce_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return default


def _coerce_bool(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("true", "1", "yes")


def compute_style_embeddings_clip(texts: list[str], device: str = "cpu") -> np.ndarray:
    """Compute CLIP text embeddings for a list of strings (text-only, no images)."""
    import torch
    import clip

    logger.info("Loading CLIP model ViT-B/32 on %s...", device)
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    truncated = []
    for t in texts:
        t = (t or " ").strip()
        if len(t) > MAX_DOCUMENT_CHARS:
            t = t[:MAX_DOCUMENT_CHARS].rsplit(" ", 1)[0] or t[:MAX_DOCUMENT_CHARS]
        truncated.append(t or " ")

    embeddings_list = []
    batch_size = 32
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i : i + batch_size]
        with torch.no_grad():
            tokens = clip.tokenize(batch, truncate=True).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings_list.append(features.cpu().float().numpy())
        if (i + batch_size) % 64 == 0 or i + batch_size >= len(truncated):
            logger.info("Computed embeddings %d/%d.", min(i + batch_size, len(truncated)), len(truncated))

    return np.vstack(embeddings_list)


def compute_style_embeddings_tfidf(texts: list[str]) -> np.ndarray:
    """Fallback: TF-IDF embeddings (no torch/CLIP). Use for --quick-test."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    vec = TfidfVectorizer(max_features=256, strip_accents="unicode", lowercase=True)
    X = vec.fit_transform([t or " " for t in texts])
    normalizer = Normalizer(norm="l2")
    return normalizer.fit_transform(X).toarray().astype(np.float32)


def compute_metadata_features(df: pd.DataFrame) -> np.ndarray:
    """Build clustering features from structured metadata only (data-backed, no parsed text).
    Uses: price_numeric, rating, review_count, price_band, is_sale, is_new, best_seller, in_stock.
    """
    from sklearn.preprocessing import StandardScaler

    price = df["price_numeric"].replace(0, np.nan).fillna(df["price_numeric"].median()).values
    price_log = np.log1p(np.maximum(price, 0.01))  # avoid log(0)
    rating = df["rating"].fillna(0).values
    rev = df["rating_count"].fillna(0).values
    review_log = np.log1p(rev)
    price_band_ord = df["price_band"].map(
        {b: i for i, b in enumerate(PRICE_BAND_LABELS + ["unknown"])}
    ).fillna(-1).astype(int).values.astype(float)
    is_sale = df["is_sale"].astype(float).values
    is_new = df["is_new"].astype(float).values
    best_seller = df["best_seller"].astype(float).values
    in_stock = df["in_stock"].astype(float).values

    X = np.column_stack([
        price_log, rating, review_log, price_band_ord,
        is_sale, is_new, best_seller, in_stock,
    ])
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32)


def _normalize_product(p: dict) -> dict:
    """Map canonical/normalized schema to internal schema used by analysis.
    Supports: average_rating→rating, review_count→rating_count, best_seller_flag→best_seller,
    new_arrival_flag→is_new, name→title, product_url→url, color_options→colors, price_original→price_regular.
    """
    out = dict(p)
    # Rating
    if "rating" not in out or out.get("rating") in (None, "", 0):
        out["rating"] = out.get("average_rating") or out.get("rating")
    if "rating_count" not in out or out.get("rating_count") in (None, "", 0):
        out["rating_count"] = out.get("review_count") or out.get("rating_count")
    # Booleans
    if "best_seller" not in out:
        out["best_seller"] = out.get("best_seller_flag")
    if "is_new" not in out:
        out["is_new"] = out.get("new_arrival_flag")
    # Title/name, url
    if not out.get("title"):
        out["title"] = out.get("name") or out.get("title")
    if not out.get("url"):
        out["url"] = out.get("product_url") or out.get("url")
    # Colors
    if not out.get("colors"):
        co = out.get("color_options")
        if isinstance(co, str):
            out["colors"] = co
        elif isinstance(co, list):
            out["colors"] = "|".join(str(x) for x in co) if co else ""
    # Price fallback
    if not out.get("price_regular") and out.get("price_original"):
        out["price_regular"] = out.get("price_original")
    # In-stock from availability_text (Chroma/canonical schema)
    if "in_stock" not in out or out.get("in_stock") is None:
        at = str(out.get("availability_text") or "").lower()
        out["in_stock"] = "in stock" in at or "available" in at
    return out


def assign_price_band(product: dict) -> str:
    """Assign price band; use price_regular/sale_price if price_current is 0."""
    p = _coerce_float(product.get("price_current"))
    if p <= 0:
        p = _coerce_float(product.get("price_regular")) or _coerce_float(product.get("sale_price"))
    if p <= 0:
        return "unknown"
    for (lo, hi), label in zip(PRICE_BANDS, PRICE_BAND_LABELS):
        if lo <= p < hi:
            return label
    return PRICE_BAND_LABELS[-1]


def assign_rating_quadrant(rating: float, rating_count: float, review_high_threshold: int) -> str:
    """Proven winner / Hidden opportunity / Market mismatch / Weak."""
    high_rating = rating >= RATING_HIGH_THRESHOLD
    high_reviews = rating_count >= review_high_threshold
    if high_rating and high_reviews:
        return "Proven winner"
    if high_rating and not high_reviews:
        return "Hidden opportunity"
    if not high_rating and high_reviews:
        return "Market mismatch"
    return "Weak"


def parse_colors(colors_str: str) -> tuple[int, str]:
    """Return (color_count, primary_color_family). 'plus N more' not counted as literal color."""
    if not colors_str or not str(colors_str).strip():
        return 0, "unknown"
    s = str(colors_str).strip()
    # Split by | or " and "
    parts = re.split(r"\s*\|\s*|\s+and\s+", s, flags=re.I)
    count = 0
    families = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.match(r"plus\s+\d+\s+more", p, re.I):
            continue
        count += 1
        low = p.lower()
        family = "unknown"
        for key, fam in COLOR_FAMILY_MAP.items():
            if key in low or low in key:
                family = fam
                break
        families.append(family)
    primary = families[0] if families else "unknown"
    return count, primary


def extract_title_keywords(title: str) -> dict[str, bool]:
    """Binary flags for TITLE_KEYWORDS (case-insensitive word boundary)."""
    t = (title or "").lower()
    return {kw: bool(re.search(r"\b" + re.escape(kw) + r"\b", t)) for kw in TITLE_KEYWORDS}


def _compute_coverage_gaps(
    df: pd.DataFrame, brand_compare: str
) -> tuple[str, list[str]]:
    """Compute coverage gaps (price band × style segment). Returns (hero_line, report_lines)."""
    if "brand" not in df.columns or "price_numeric" not in df.columns:
        return "", []
    brand_vals = df["brand"].fillna("").astype(str).str.strip()
    is_target = brand_vals.str.lower() == brand_compare.lower()
    if is_target.sum() == 0 or (~is_target).sum() == 0:
        return "", []

    # Build text for segment detection (name/title + description + category)
    text_parts = []
    for col in ("name", "title", "description", "category_breadcrumb"):
        if col in df.columns:
            text_parts.append(df[col].fillna("").astype(str))
    text = pd.Series("", index=df.index)
    if text_parts:
        text = text_parts[0]
        for p in text_parts[1:]:
            text = text + " " + p
    text = text.str.lower()

    # Price band (includes $60–90 sweet spot)
    price = df["price_numeric"].fillna(0)
    df_cov = df.copy()
    df_cov["_band"] = "unknown"
    for i, (lo, hi) in enumerate(COVERAGE_PRICE_BANDS):
        mask = (price >= lo) & (price < hi)
        df_cov.loc[mask, "_band"] = COVERAGE_BAND_LABELS[i]
    df_cov["_text"] = text
    df_cov["_is_tote"] = text.str.contains("tote", na=False, regex=False)
    df_cov["_is_work"] = text.str.contains(
        r"work|laptop|professional|office", na=False, regex=True
    )
    df_cov["_segment"] = "other"
    df_cov.loc[
        df_cov["_is_tote"] & df_cov["_is_work"], "_segment"
    ] = "work_tote"
    df_cov.loc[
        df_cov["_is_tote"] & ~df_cov["_is_work"], "_segment"
    ] = "tote"

    # Filter valid band
    df_cov = df_cov[df_cov["_band"] != "unknown"]
    if df_cov.empty:
        return "", []

    # Group: band × segment × brand
    df_cov["_group"] = np.where(is_target, brand_compare, "Others")
    ct = df_cov.groupby(["_band", "_segment", "_group"]).size().unstack(
        fill_value=0
    )
    gap_col = f"{brand_compare}"
    others_col = "Others"
    if gap_col not in ct.columns:
        gap_col = ct.columns[0]
    if others_col not in ct.columns:
        others_col = [c for c in ct.columns if c != gap_col]
        others_col = others_col[0] if others_col else None
    if others_col is None:
        return "", []

    ct["gap_skus"] = ct[others_col] - ct[gap_col]
    gaps = ct[ct["gap_skus"] > 0].sort_values("gap_skus", ascending=False)

    # Hero insight
    hero = ""
    report_lines = []
    if len(gaps) > 0:
        top = gaps.iloc[0]
        idx = gaps.index[0]
        band, seg = idx[0], idx[1]
        n_gap = int(top[gap_col])
        n_others = int(top[others_col])
        seg_label = (
            "structured work totes"
            if seg == "work_tote"
            else ("totes" if seg == "tote" else seg.replace("_", " "))
        )
        hero = (
            f"Coverage gap of {int(top['gap_skus'])} SKUs in ${band} {seg_label} "
            f"versus category leader ({n_others} vs {n_gap})."
        )
        report_lines = [
            "",
            "**Why this matters:**",
            "- Mid-tier ($60–90) = sweet spot for AOV",
            "- Work totes = office return + higher basket",
            "- Structured = aspirational positioning",
            "- SKU gap = directly actionable buy decision",
            "",
            "**All coverage gaps (Others ahead):**",
            "",
        ]
        for idx, row in gaps.head(10).iterrows():
            band, seg = idx[0], idx[1]
            seg_label = (
                "work totes"
                if seg == "work_tote"
                else ("totes" if seg == "tote" else seg)
            )
            report_lines.append(
                f"- ${band} {seg_label}: {int(row['gap_skus'])} SKU gap "
                f"({int(row[others_col])} Others vs {int(row[gap_col])} {brand_compare})"
            )
    else:
        # No gaps – show closest (smallest Gap lead)
        ct["lead"] = ct[gap_col] - ct[others_col]
        close = ct[ct["lead"] >= 0].sort_values("lead", ascending=True)
        if len(close) > 0:
            top = close.iloc[0]
            idx = close.index[0]
            band, seg = idx[0], idx[1]
            seg_label = (
                "work totes"
                if seg == "work_tote"
                else ("totes" if seg == "tote" else seg)
            )
            hero = (
                f"No coverage gaps in current data. Closest segment: ${band} {seg_label} "
                f"({int(top[gap_col])} {brand_compare} vs {int(top[others_col])} Others)."
            )
        else:
            hero = "No coverage gaps identified with current data."
    return hero, report_lines


def run_gap_analysis(
    output_dir: Path,
    input_path: Path | None = None,
    products: list[dict] | None = None,
    src_label: str | None = None,
    n_clusters: int = 5,
    run_hdbscan: bool = True,
    run_umap: bool = True,
    review_high_threshold: int | None = None,
    quick_test: bool = False,
    plot_umap: bool = False,
    umap_color_by: str = "cluster_kmeans",
    brand_compare: str | None = "Gap",
    use_metadata: bool = False,
    balance_clustering: bool = False,
) -> None:
    """Load data (from file or Chroma), embed, cluster, analyze, write enriched data and report."""
    if products is not None:
        src_label = src_label or "ChromaDB"
    elif input_path:
        products = load_products(input_path)
        src_label = src_label or str(input_path)
    else:
        raise ValueError("Provide input_path or products (e.g. --from-chroma).")
    products = [_normalize_product(p) for p in products]
    logger.info("Loaded %d products from %s", len(products), src_label)
    if not products:
        logger.error("No products to analyze.")
        return

    # Build DataFrame first (needed for metadata features and balance sampling)
    df = pd.DataFrame(products)
    df["price_band"] = [assign_price_band(p) for p in products]
    price_current_num = [_coerce_float(p.get("price_current")) or _coerce_float(p.get("price_regular")) or _coerce_float(p.get("sale_price")) for p in products]
    df["price_numeric"] = price_current_num
    ratings = [_coerce_float(p.get("rating")) for p in products]
    rating_counts = [_coerce_float(p.get("rating_count")) for p in products]
    df["rating"] = ratings
    df["rating_count"] = rating_counts
    if review_high_threshold is None:
        review_high_threshold = int(np.median(rating_counts)) if rating_counts else REVIEW_COUNT_HIGH_DEFAULT
    df["rating_quadrant"] = [
        assign_rating_quadrant(r, rc, review_high_threshold) for r, rc in zip(ratings, rating_counts)
    ]
    color_parsed = [parse_colors(p.get("colors")) for p in products]
    df["color_count"] = [c for c, _ in color_parsed]
    df["color_family"] = [f for _, f in color_parsed]
    for kw in TITLE_KEYWORDS:
        col = "kw_" + kw.replace(" ", "_")
        df[col] = [extract_title_keywords(p.get("title") or "").get(kw, False) for p in products]
    df["is_sale"] = [_coerce_bool(p.get("is_sale")) for p in products]
    df["is_clearance"] = [_coerce_bool(p.get("is_clearance")) for p in products]
    df["is_new"] = [_coerce_bool(p.get("is_new")) for p in products]
    df["best_seller"] = [_coerce_bool(p.get("best_seller")) for p in products]
    df["in_stock"] = [_coerce_bool(p.get("in_stock")) for p in products]

    # Clustering features: metadata (reliable) or text embeddings (can be noisy)
    style_texts = [build_style_text(p) for p in products]
    df["style_text"] = style_texts
    if use_metadata:
        logger.info("Using metadata-based features (structured fields only, no parsed text).")
        features = compute_metadata_features(df)
    else:
        if quick_test:
            logger.info("Quick-test mode: using TF-IDF embeddings (no CLIP).")
            features = compute_style_embeddings_tfidf(style_texts)
        else:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                raise ImportError(
                    "Gap analysis requires PyTorch and CLIP for style embeddings. "
                    "Install with: pip install torch; pip install git+https://github.com/openai/CLIP.git. "
                    "Or run with --quick-test or --use-metadata."
                ) from None
            features = compute_style_embeddings_clip(style_texts, device=device)
    logger.info("Computed %d feature vectors.", len(features))

    # Stratified sampling for clustering when brand-imbalanced (e.g. 160 Gap vs 50 Others)
    fit_indices = np.arange(len(df))
    if balance_clustering and brand_compare and "brand" in df.columns:
        brand_vals = df["brand"].fillna("").astype(str).str.strip()
        is_target = brand_vals.str.lower() == brand_compare.lower()
        idx_gap = np.where(is_target)[0]
        idx_others = np.where(~is_target)[0]
        n_gap, n_others = len(idx_gap), len(idx_others)
        if n_gap > 0 and n_others > 0 and abs(n_gap - n_others) > 10:
            n_sample = min(n_gap, n_others)
            rng = np.random.default_rng(42)
            sample_gap = rng.choice(idx_gap, size=min(n_sample, n_gap), replace=False)
            sample_others = rng.choice(idx_others, size=min(n_sample, n_others), replace=False)
            fit_indices = np.concatenate([sample_gap, sample_others])
            logger.info("Balanced clustering: fit on %d stratified samples (Gap=%d, Others=%d).", len(fit_indices), len(sample_gap), len(sample_others))

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features[fit_indices])
    df["cluster_kmeans"] = kmeans.predict(features)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(features)
    df["pca_x"] = pca_2d[:, 0]
    df["pca_y"] = pca_2d[:, 1]

    if run_hdbscan:
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
            df["cluster_hdbscan"] = clusterer.fit_predict(features)
        except Exception as e:
            logger.warning("HDBSCAN failed: %s. Skipping.", e)
            df["cluster_hdbscan"] = -1
    else:
        df["cluster_hdbscan"] = -1

    if run_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features) - 1))
            umap_2d = reducer.fit_transform(features)
            df["umap_x"] = umap_2d[:, 0]
            df["umap_y"] = umap_2d[:, 1]
        except Exception as e:
            logger.warning("UMAP failed: %s. Skipping.", e)
            df["umap_x"] = np.nan
            df["umap_y"] = np.nan
    else:
        df["umap_x"] = np.nan
        df["umap_y"] = np.nan

    cluster_col = "cluster_kmeans"

    # Write outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"gap_analysis_handbags_{ts}"

    enriched_csv = output_dir / f"{base}.csv"
    enriched_json = output_dir / f"{base}.json"
    report_md = output_dir / f"{base}_report.md"
    umap_png = output_dir / f"{base}_umap.png"

    # Optional: UMAP plot (generate before report so report can embed it)
    umap_plot_written = False
    if plot_umap:
        if not run_umap:
            raise ValueError("--plot-umap requested but UMAP was disabled (use without --no-umap).")
        try:
            save_umap_scatter(
                df,
                out_path=umap_png,
                color_by=umap_color_by,
                title=f"UMAP: {len(df)} products (colored by {umap_color_by})",
            )
            umap_plot_written = True
            logger.info("Wrote %s", umap_png)
        except Exception as e:
            logger.warning("UMAP plot failed: %s", e)

    # --- Build report sections ---
    report_lines = [
        "# Gap Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Input: {src_label}",
        f"Products: {len(df)}",
        f"Outputs: {enriched_csv.name}, {enriched_json.name}",
        "",
        "---",
        "",
    ]

    # Hero: Coverage gap (show first)
    has_hero = False
    if brand_compare and "brand" in df.columns:
        hero, cov_lines = _compute_coverage_gaps(df, brand_compare)
        if hero:
            has_hero = True
            report_lines.extend([
                "## 1. Coverage gap in core professional price band",
                "",
                hero,
            ])
            report_lines.extend(cov_lines)
            report_lines.extend(["", "---", "", "## 2. Visual evidence", ""])
        else:
            report_lines.extend(["## 1. Visual evidence", ""])
    else:
        report_lines.extend(["## 1. Visual evidence", ""])

    n_summary = 3 if has_hero else 2
    n_brand = 4 if has_hero else 3
    n_style = 5 if has_hero else 4

    report_lines.extend([
        "",
        (f"UMAP scatter (colored by `{umap_color_by}`): ![]({umap_png.name})" if umap_plot_written else "UMAP scatter: (not generated)"),
        "",
        "---",
        "",
        f"## {n_summary}. Summary",
        "",
        f"- **Products:** {len(df)}",
        f"- **Clustering:** {'metadata (structured fields)' if use_metadata else 'text embeddings (CLIP/TF-IDF)'}",
        f"- **Style clusters (K-Means k={n_clusters}):** {df[cluster_col].nunique()}",
        f"- **Price bands:** {df['price_band'].nunique()} ({', '.join(PRICE_BAND_LABELS)} + unknown)",
        "",
        "---",
        "",
    ])

    # --- Gap vs Others brand comparison ---
    if brand_compare and "brand" in df.columns:
        brand_col = "brand"
        brand_vals = df[brand_col].fillna("").astype(str).str.strip()
        is_target = brand_vals.str.lower() == brand_compare.lower()
        gap_df = df[is_target]
        others_df = df[~is_target]
        n_gap = len(gap_df)
        n_others = len(others_df)
        report_lines.extend([
            f"## {n_brand}. Brand comparison: {brand_compare} vs Others",
            "",
        ])
        if n_gap == 0:
            report_lines.append(f"**Warning:** No {brand_compare} products in this dataset. "
                "Run with `--from-chroma` to use your ChromaDB data (160 Gap, 50 Others).")
            report_lines.append("")
        elif abs(n_gap - n_others) > 20:
            report_lines.append(f"*Note: Imbalanced sample ({brand_compare}={n_gap}, Others={n_others}). "
                "Per-SKU metrics below are comparable; use `--balance-clustering` to reduce cluster bias from majority brand.*")
            report_lines.append("")
        report_lines.extend([
            f"| Metric | {brand_compare} | Others |",
            "|--------|--------|--------|",
            f"| **SKU count** | {n_gap} | {n_others} |",
        ])
        # Avg price
        avg_gap = gap_df["price_numeric"].replace(0, np.nan).mean()
        avg_oth = others_df["price_numeric"].replace(0, np.nan).mean()
        def _fmt(v, fmt: str = ".1f"):
            if v is None or (isinstance(v, (int, float)) and np.isnan(v)):
                return "—"
            return f"{v:{fmt}}"
        report_lines.append(f"| **Avg price** | {_fmt(avg_gap)} | {_fmt(avg_oth)} |")
        # Avg rating
        avg_r_gap = gap_df["rating"].replace(0, np.nan).mean()
        avg_r_oth = others_df["rating"].replace(0, np.nan).mean()
        report_lines.append(f"| **Avg rating** | {_fmt(avg_r_gap, '.2f')} | {_fmt(avg_r_oth, '.2f')} |")
        # Sale %
        pct_sale_gap = f"{gap_df['is_sale'].mean() * 100:.0f}%" if n_gap else "—"
        pct_sale_oth = f"{others_df['is_sale'].mean() * 100:.0f}%" if n_others else "—"
        report_lines.append(f"| **Sale %** | {pct_sale_gap} | {pct_sale_oth} |")
        # Best seller %
        pct_best_gap = f"{gap_df['best_seller'].mean() * 100:.0f}%" if n_gap else "—"
        pct_best_oth = f"{others_df['best_seller'].mean() * 100:.0f}%" if n_others else "—"
        report_lines.append(f"| **Best seller %** | {pct_best_gap} | {pct_best_oth} |")
        # New arrival %
        pct_new_gap = f"{gap_df['is_new'].mean() * 100:.0f}%" if n_gap else "—"
        pct_new_oth = f"{others_df['is_new'].mean() * 100:.0f}%" if n_others else "—"
        report_lines.append(f"| **New arrival %** | {pct_new_gap} | {pct_new_oth} |")
        # In-stock %
        pct_stock_gap = f"{gap_df['in_stock'].mean() * 100:.0f}%" if n_gap else "—"
        pct_stock_oth = f"{others_df['in_stock'].mean() * 100:.0f}%" if n_others else "—"
        report_lines.append(f"| **In-stock %** | {pct_stock_gap} | {pct_stock_oth} |")
        # Price band distribution
        report_lines.extend(["", "**Price band distribution:**", ""])
        for band in PRICE_BAND_LABELS + ["unknown"]:
            c_gap = (gap_df["price_band"] == band).sum()
            c_oth = (others_df["price_band"] == band).sum()
            report_lines.append(f"- {band}: {brand_compare}={c_gap}, Others={c_oth}")
        # Cluster distribution
        report_lines.extend(["", "**Cluster distribution:**", ""])
        for c in sorted(df[cluster_col].unique()):
            c_gap = (gap_df[cluster_col] == c).sum()
            c_oth = (others_df[cluster_col] == c).sum()
            report_lines.append(f"- Cluster {c}: {brand_compare}={c_gap}, Others={c_oth}")
        report_lines.extend(["", "---", ""])

    report_lines.extend([
        f"## {n_style}. Style clusters (SKU count, avg price, avg rating, dominant brand)",
        "",
    ])

    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        n = len(sub)
        avg_price = sub["price_numeric"].replace(0, np.nan).mean()
        avg_price_str = f"{avg_price:.1f}" if not np.isnan(avg_price) else "—"
        avg_rating = sub["rating"].replace(0, np.nan).mean()
        avg_rating_str = f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "—"
        top_brand = sub["brand"].mode().iloc[0] if "brand" in sub.columns and len(sub["brand"].mode()) else "—"
        report_lines.append(f"- **Cluster {c}:** SKUs={n}, avg_price={avg_price_str}, avg_rating={avg_rating_str}, top_brand={top_brand}")
    report_lines.extend(["", "---", "", f"## {n_style + 1}. SKU density (cluster × price band)", ""])

    # Pivot (use string table; to_markdown requires tabulate)
    pivot = pd.crosstab(df[cluster_col], df["price_band"], margins=True)
    report_lines.append("```")
    report_lines.append(pivot.to_string())
    report_lines.append("```")
    report_lines.extend(["", "*(Empty or near-zero cells = price–style gap)*", "", "---", "", f"## {n_style + 2}. Rating × review quadrant", ""])

    quad_counts = df["rating_quadrant"].value_counts()
    for q, cnt in quad_counts.items():
        report_lines.append(f"- **{q}:** {cnt}")
    report_lines.extend(["", "Hidden winners (high rating, low reviews) by cluster:", ""])
    hidden = df[df["rating_quadrant"] == "Hidden opportunity"].groupby(cluster_col).size()
    for c, cnt in hidden.items():
        report_lines.append(f"- Cluster {c}: {cnt}")
    report_lines.extend(["", "---", "", f"## {n_style + 3}. Brand dominance (per cluster)", ""])

    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        brand_counts = sub["brand"].value_counts()
        total = len(sub)
        top = brand_counts.iloc[0] if len(brand_counts) else 0
        pct = 100 * top / total if total else 0
        report_lines.append(f"- **Cluster {c}:** top brand share = {pct:.0f}% — {'dominance' if pct >= 70 else 'mixed/white-space'}")
    # Sections 7 & 8 use parsed text (colors, title keywords) — skip when metadata-only mode
    if not use_metadata:
        report_lines.extend(["", "---", "", f"## {n_style + 4}. Color diversity (by cluster)", ""])
        for c in sorted(df[cluster_col].unique()):
            sub = df[df[cluster_col] == c]
            report_lines.append(f"- **Cluster {c}:** avg color_count={sub['color_count'].mean():.1f}, main families: {sub['color_family'].value_counts().head(3).to_dict()}")
        report_lines.extend(["", "---", "", f"## {n_style + 5}. Title keyword co-occurrence (gaps)", ""])
        cooc = {}
        for i, ki in enumerate(TITLE_KEYWORDS):
            for j, kj in enumerate(TITLE_KEYWORDS):
                if i < j:
                    pair = (ki, kj)
                    count = ((df["kw_" + ki.replace(" ", "_")] == True) & (df["kw_" + kj.replace(" ", "_")] == True)).sum()
                    cooc[pair] = count
        sorted_pairs = sorted(cooc.items(), key=lambda x: -x[1])
        gap_pairs = [(k1, k2) for (k1, k2), c in sorted_pairs if c == 0]
        report_lines.append("Top pairs (by count):")
        for (k1, k2), cnt in sorted_pairs[:5]:
            report_lines.append(f"- {k1} + {k2}: {cnt}")
        report_lines.append("")
        report_lines.append("Gaps (count = 0):")
        for (k1, k2) in gap_pairs[:20]:
            report_lines.append(f"- {k1} + {k2}")
        if len(gap_pairs) > 20:
            report_lines.append(f"- ... and {len(gap_pairs) - 20} more")
    else:
        report_lines.extend(["", "---", "", f"## {n_style + 4}. Color / title sections (skipped)", ""])
        report_lines.append("*Skipped in metadata-only mode — color diversity and title keywords come from parsed text.*")
    gaps_num = n_style + (5 if use_metadata else 6)
    report_lines.extend(["", "---", "", f"## {gaps_num}. Six gap analyses", ""])

    # Rating-weighted cluster strength
    df["score_weight"] = df["rating"] * np.log(df["rating_count"] + 1)
    cluster_score = df.groupby(cluster_col).agg(cluster_score=("score_weight", "mean"))
    cluster_score["sku_count"] = df.groupby(cluster_col).size()
    report_lines.append(f"### {gaps_num}.1 Rating-weighted cluster strength")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(cluster_score.sort_values("cluster_score", ascending=False).to_string())
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("- High score + low SKU count → expand. Low score + high SKU count → over-indexed.")
    report_lines.extend([f"", f"### {gaps_num}.2 New vs established", ""])
    new_by_cluster = df.groupby(cluster_col)["is_new"].sum()
    report_lines.append("```")
    report_lines.append(new_by_cluster.to_string())
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("- If new SKUs concentrated in one cluster → not innovating in other styles.")
    report_lines.extend([f"", f"### {gaps_num}.3 Bestseller gap", ""])
    best = df[df["best_seller"] == True]
    if len(best):
        report_lines.append(f"Bestsellers in clusters: {best[cluster_col].value_counts().to_dict()}")
        report_lines.append("")
        report_lines.append("- If bestseller cluster has few total SKUs → expand in that direction.")
    else:
        report_lines.append("No best_seller=True in dataset.")
    report_lines.extend([f"", f"### {gaps_num}.4 Clearance pattern", ""])
    clearance_pct = df.groupby(cluster_col)["is_clearance"].mean() * 100
    report_lines.append("```")
    report_lines.append(clearance_pct.to_string())
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("- High clearance % in a cluster → possible weak demand.")
    report_lines.extend([f"", f"### {gaps_num}.5 In-stock pressure", ""])
    out_of_stock_pct = (1 - df.groupby(cluster_col)["in_stock"].mean()) * 100
    report_lines.append("```")
    report_lines.append(out_of_stock_pct.to_string())
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("- High out-of-stock % → demand signal.")
    report_lines.extend(["", "---", "", "*Internal trend map; competitor comparison when data available.*", ""])

    report_text = "\n".join(report_lines)

    # CSV: convert bool to int for cleaner export if needed; keep as bool
    df.to_csv(enriched_csv, index=False)
    logger.info("Wrote %s", enriched_csv)

    # JSON: list of dicts (drop numpy types)
    records = df.replace({np.nan: None}).to_dict("records")
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, (np.integer, np.floating)):
                r[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.bool_):
                r[k] = bool(v)
    with open(enriched_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info("Wrote %s", enriched_json)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Wrote %s", report_md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gap analysis: clustering, price/rating/brand/color/co-occurrence, six gap analyses."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Path to JSON, JSONL, or CSV (ignored if --from-chroma)",
    )
    parser.add_argument(
        "--from-chroma",
        action="store_true",
        help="Load products from ChromaDB instead of file (requires CHROMA_API_KEY)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="target_handbags",
        help="Chroma collection name when using --from-chroma",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Output directory for enriched data and report",
    )
    parser.add_argument(
        "-k", "--clusters",
        type=int,
        default=5,
        help="K-Means n_clusters (default 5)",
    )
    parser.add_argument(
        "--no-hdbscan",
        action="store_true",
        help="Skip HDBSCAN",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP",
    )
    parser.add_argument(
        "--review-high",
        type=int,
        default=None,
        help="Review count threshold for 'high' in quadrant (default: median)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use TF-IDF embeddings instead of CLIP (no torch required)",
    )
    parser.add_argument(
        "--plot-umap",
        action="store_true",
        help="Save a UMAP scatter plot PNG in the output directory",
    )
    parser.add_argument(
        "--umap-color-by",
        type=str,
        default="cluster_kmeans",
        help="Column to color UMAP points by (default: cluster_kmeans). Examples: price_band, rating_quadrant, brand",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="Gap",
        help="Brand to compare vs Others (default: Gap). Use '' to skip brand comparison.",
    )
    parser.add_argument(
        "--use-metadata",
        action="store_true",
        help="Cluster on structured metadata only (price, rating, flags) — more reliable than parsed text",
    )
    parser.add_argument(
        "--balance-clustering",
        action="store_true",
        help="Stratify K-Means fit when brand-imbalanced (e.g. 160 Gap vs 50 Others)",
    )
    args = parser.parse_args()

    if args.from_chroma:
        products = load_products_from_chroma(collection_name=args.collection)
        input_path = None
    else:
        input_path = args.input or (PROJECT_ROOT / "output" / "target_handbags_20260220_213726.csv")
        products = None

    run_gap_analysis(
        output_dir=args.output_dir,
        input_path=input_path,
        products=products,
        n_clusters=args.clusters,
        run_hdbscan=not args.no_hdbscan,
        run_umap=not args.no_umap,
        review_high_threshold=args.review_high,
        quick_test=args.quick_test,
        plot_umap=args.plot_umap,
        umap_color_by=args.umap_color_by,
        brand_compare=args.brand if args.brand else None,
        use_metadata=args.use_metadata,
        balance_clustering=args.balance_clustering,
    )


if __name__ == "__main__":
    main()
