#!/usr/bin/env python3
"""
Enhanced taxonomy clustering: cleaner signal, hybrid embeddings, HDBSCAN, auto discovery.

Upgrades over run_taxonomy_clustering.py:
  1. Brand tokens from CSV brand column + statistical stopwords (>80% doc freq)
  2. Hybrid embeddings: MiniLM/TF-IDF + TF-IDF n-grams + price/material/sale flags
  3. HDBSCAN within each silhouette (no k guessing, handles noise)
  4. Evaluation: silhouette, Davies-Bouldin, retailer purity, price variance
  5. Optional stronger model: all-mpnet-base-v2

Visual upgrades (NO change to clustering logic):
  - Keep PCA scatter (existing), but add optional UMAP/t-SNE embeddings for visualization
  - Add per-silhouette 2D plots (reduces clutter)
  - Add density-ish view via alpha + downsampling
  - Add “top segments only” plots

Usage:
  python scripts/run_taxonomy_clustering_enhanced.py -n output/products_normalized_combined.csv
  python scripts/run_taxonomy_clustering_enhanced.py --tfidf --min-cluster-size 10
  python scripts/run_taxonomy_clustering_enhanced.py --viz umap
  python scripts/run_taxonomy_clustering_enhanced.py --viz tsne
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional viz libs (do not affect clustering)
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

logging = __import__("logging")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "output" / "products_normalized_combined.csv"
DEFAULT_OUT = PROJECT_ROOT / "output" / "taxonomy_enhanced"

# Base stopwords (always remove)
BASE_STOP = frozenset({
    "the", "and", "for", "with", "this", "that", "your", "more", "made", "everything",
    "one", "women", "accessories", "accessory", "bag", "bags", "handbag", "handbags",
    "purse", "purses", "clothing", "shoes", "size", "our", "you", "from", "into",
    "also", "have", "can", "get", "target", "gap",
})

# Overused structural words - remove to expose shape/material differences
STRUCTURAL_WORDS = frozenset({"zip", "strap", "pockets", "pocket", "closure", "compartments", "adjustable"})

_VIEWED_LIKE = re.compile(
    r"(?:Shipping\s*&\s*returns\s+)?(?:Customers\s+also\s+viewed|You\s+may\s+also\s+like).*",
    re.I | re.S,
)
_PROMO = [
    re.compile(r"\d+%\s*off\s*\+\s*extra\s+10%\s*in\s+the\s+app", re.I),
    re.compile(r"(?:extra\s+)?\d+%\s*off?\s*in\s+the\s+app", re.I),
    re.compile(r"\$\d+\.?\d*(?:\s+\$\d+\.?\d*)?", re.I),
]
_SPEC = [
    (re.compile(r"\([HWDhwd]\)", re.I), " "),
    (re.compile(r"\b\d+\s*Inches?\s*\(\s*[HWD]\s*\)", re.I), " "),
]

SIL_RULES = [
    ("accessory", r"\b(charm|keychain|key\s*ring|organizer|organiser|pouch|coin\s*purse)\b"),
    ("backpack", r"\b(backpack|rucksack)\b"),
    ("clutch", r"\b(clutch|wristlet|evening)\b"),
    ("tote", r"\b(tote|shopper)\b"),
    ("bucket", r"\b(bucket|drawstring)\b"),
    ("satchel", r"\b(satchel|top\s*handle|structured\s*bag)\b"),
    ("crossbody", r"\b(crossbody|cross\s*body|camera\s*bag)\b"),
    ("shoulder", r"\b(shoulder)\b"),
    ("mini", r"\b(mini|micro)\b"),
    ("messenger", r"\b(messenger)\b"),
    ("weekender", r"\b(weekender|duffel)\b"),
]


def _strip_promo(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = _VIEWED_LIKE.sub(" ", text)
    for p in _PROMO:
        t = p.sub(" ", t)
    for pat, repl in _SPEC:
        t = pat.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()


def _find(pat: str, s: str) -> bool:
    return bool(re.search(pat, s, re.I))


def extract_material(s: str) -> str:
    s = (s or "").lower()
    if _find(r"\bvegan\s*leather\b", s):
        return "vegan_leather"
    if _find(r"\bleather\b", s):
        return "leather"
    if _find(r"\bnylon\b", s):
        return "nylon"
    if _find(r"\b(canvas|cotton)\b", s):
        return "canvas_cotton"
    if _find(r"\bstraw\b", s):
        return "straw"
    if _find(r"\brecycled\b", s):
        return "recycled"
    return "unknown"


def classify_silhouette(raw: str) -> str:
    s = (raw or "").lower()
    for label, pat in SIL_RULES:
        if re.search(pat, s):
            return label
    return "other"


def build_raw_text(row: dict) -> str:
    """Build text for embeddings. Prioritizes name, leaf_category, material for semantic signal."""
    def _s(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v).strip()
        return "" if s in ("", "nan", "None", "{}") else s

    # Core fields for embedding (name, leaf_category, material first)
    name = _s(row.get("name") or row.get("product_name"))
    leaf_cat = _s(row.get("leaf_category"))
    material = _s(row.get("material_text")) or _s(row.get("materials_section"))

    parts = [name, leaf_cat, material]
    parts.extend([
        _s(row.get("description")),
        _s(row.get("product_details")),
        _s(row.get("category_breadcrumb") or row.get("category_breadcrumbs")),
        _s(row.get("bag_structure")),
        _s(row.get("interior_features")),
        _s(row.get("exterior_features")),
        _s(row.get("closure_type")),
        _s(row.get("handle_type")),
        _s(row.get("fabric_name")),
    ])
    return " ".join(p for p in parts if p)


SILHOUETTE_MATERIAL_WHITELIST = frozenset({
    "tote", "crossbody", "shoulder", "clutch", "backpack", "satchel", "bucket",
    "weekender", "messenger", "mini", "charm", "keychain", "organizer", "pouch",
    "leather", "nylon", "canvas", "cotton", "vegan", "recycled", "straw",
})


def get_brand_tokens_from_column(df: pd.DataFrame, brand_col: str = "brand") -> frozenset:
    """
    Extract brand tokens from the CSV brand column (actual brands, may have multiple per row).
    Excludes silhouette/material terms. Handles multi-word brands like 'Vera Bradley', 'A New Day'.
    """
    skip = BASE_STOP | SILHOUETTE_MATERIAL_WHITELIST | {
        "new", "day", "good", "gather", "shade", "shore", "up", "universal", "thread",
    }
    tokens = set()
    if brand_col not in df.columns:
        return frozenset()
    for val in df[brand_col].dropna().astype(str).unique():
        if not val or val.strip().lower() in ("nan", "none", ""):
            continue
        words = re.findall(r"\b[a-z]{2,}\b", val.lower())
        for w in words:
            if len(w) > 2 and w not in skip:
                tokens.add(w)
    return frozenset(tokens)


def detect_statistical_stopwords(
    texts: list[str],
    min_df: int = 10,
    doc_freq_threshold: float = 0.80,
) -> frozenset:
    """Words appearing in >threshold% of docs (ultra-frequent, low discriminative power)."""
    cv = CountVectorizer(min_df=min_df)
    try:
        X = cv.fit_transform(texts)
    except Exception:
        return frozenset()
    n_docs = X.shape[0]
    doc_freq = np.array((X > 0).sum(axis=0)).ravel()
    words = cv.get_feature_names_out()
    high = {w for w, df_ in zip(words, doc_freq) if df_ >= doc_freq_threshold * n_docs}
    return frozenset(high)


def clean_text_enhanced(
    s: str,
    brand_tokens: frozenset,
    statistical_stop: frozenset,
) -> str:
    """Aggressive cleaning: promo strip, brands, statistical stopwords, structural noise."""
    if pd.isna(s):
        return ""
    s = _strip_promo(str(s))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) > 2]
    stop = BASE_STOP | brand_tokens | statistical_stop | STRUCTURAL_WORDS
    toks = [t for t in toks if t not in stop]
    return " ".join(toks)


def get_hybrid_embeddings(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    use_sentence_transformer: bool = True,
    model_name: str = "all-MiniLM-L6-v2",
    tfidf_dim: int = 64,
    add_numeric: bool = True,
) -> np.ndarray:
    """Hybrid: (MiniLM or TF-IDF-SVD) + TF-IDF bigrams + price_bucket + material_enc + is_sale."""
    texts = df[text_col].fillna("").tolist()
    texts = [t if t.strip() else "(no description)" for t in texts]

    parts = []

    # 1) Semantic embedding
    if use_sentence_transformer:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            parts.append(emb)
        except Exception as ex:
            logger.warning("SentenceTransformer failed (%s), using TF-IDF", ex)
            use_sentence_transformer = False

    if not use_sentence_transformer:
        tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.95, ngram_range=(1, 2))
        X_t = tfidf.fit_transform(texts)
        n_comp = min(tfidf_dim, X_t.shape[0] - 1, X_t.shape[1] - 1)
        n_comp = max(2, n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        emb = svd.fit_transform(X_t)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        parts.append(emb / norms)

    # 2) TF-IDF bigrams (discriminative phrases)
    tfidf2 = TfidfVectorizer(max_features=200, min_df=2, max_df=0.9, ngram_range=(2, 2))
    X_t2 = tfidf2.fit_transform(texts)
    n2 = min(32, X_t2.shape[1] - 1, X_t2.shape[0] - 1)
    n2 = max(2, n2)
    svd2 = TruncatedSVD(n_components=n2, random_state=43)
    ph = svd2.fit_transform(X_t2)
    parts.append(ph)

    # 3) Numeric flags
    if add_numeric and len(parts) > 0:
        price = pd.to_numeric(df["price_current"], errors="coerce").fillna(0).values
        price_bucket = np.clip(np.floor(np.log1p(price) / 2).astype(int), 0, 4)
        is_sale = df["is_sale"].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0).values
        le = LabelEncoder()
        mat_enc = le.fit_transform(df["material"].fillna("unknown").astype(str))
        num = np.column_stack([
            StandardScaler().fit_transform(price_bucket.reshape(-1, 1)),
            is_sale.reshape(-1, 1),
            StandardScaler().fit_transform(mat_enc.reshape(-1, 1)),
        ])
        parts.append(num)

    X = np.hstack(parts)
    return StandardScaler().fit_transform(X)


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 10) -> np.ndarray:
    """HDBSCAN clustering; noise (-1) reassigned to nearest non-noise neighbor's cluster."""
    if not HAS_HDBSCAN:
        from sklearn.cluster import KMeans
        k = max(2, min(10, X.shape[0] // 5))
        return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(2, min_cluster_size // 3),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    noise_mask = labels == -1
    if noise_mask.sum() > 0 and (labels >= 0).sum() > 0:
        from sklearn.neighbors import NearestNeighbors
        valid_idx = np.where(~noise_mask)[0]
        nn = NearestNeighbors(n_neighbors=1).fit(X[valid_idx])
        _, nearest_in_valid = nn.kneighbors(X[noise_mask])
        labels[noise_mask] = labels[valid_idx[nearest_in_valid.ravel()]]
    return labels


def _price_tier(med: float) -> str:
    if med < 25:
        return "Entry"
    if med < 60:
        return "Mid"
    if med < 120:
        return "Premium"
    return "Luxury"


def tag_price_tier(p: float) -> str:
    """Tag individual product price into tier for analytics."""
    if p < 30:
        return "Entry"
    if p < 60:
        return "Mid"
    if p < 90:
        return "Mid+"
    if p < 120:
        return "Upper Mid"
    return "Premium"


def label_cluster_enhanced(block: pd.DataFrame, subcluster_col: str = "subcluster") -> dict[int, str]:
    if subcluster_col not in block.columns:
        return {}
    vec = TfidfVectorizer(ngram_range=(2, 3), min_df=1)
    try:
        X = vec.fit_transform(block["text_clean"].fillna(""))
    except Exception:
        return {}
    feats = np.array(vec.get_feature_names_out())
    if len(feats) == 0:
        return {}

    labels = {}
    for c in sorted(block[subcluster_col].dropna().unique()):
        c = int(c)
        if c < 0:
            continue
        idx = (block[subcluster_col] == c).values
        if idx.sum() < 2:
            labels[c] = "single"
            continue
        mean_vec = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[-3:][::-1]
        top_phrases = [feats[i] for i in top_idx if i < len(feats)]
        med_price = block.loc[idx, "price_current"].median()
        mat = block.loc[idx, "material"].mode().iloc[0] if "material" in block.columns else "unknown"
        tier = _price_tier(float(med_price))
        labels[c] = f"{mat} | {' / '.join(top_phrases[:2])} | {tier}"
    return labels


def evaluate_clusters(df: pd.DataFrame, X: np.ndarray, label_col: str = "subcluster") -> dict:
    """Silhouette, Davies-Bouldin, retailer purity, price variance."""
    labels = df[label_col].values
    valid = labels >= 0
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"silhouette": 0.0, "davies_bouldin": 999.0, "purity_avg": 0.0, "price_var_avg": 0.0}

    sil = float(silhouette_score(X[valid], labels[valid]))
    db = float(davies_bouldin_score(X[valid], labels[valid]))

    purities = []
    price_vars = []
    for c in np.unique(labels[valid]):
        mask = labels == c
        sub = df[mask]
        maj = sub["source"].value_counts().max()
        purities.append(maj / len(sub))
        price_vars.append(sub["price_current"].var())
    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "purity_avg": float(np.mean(purities)) if purities else 0.0,
        "price_var_avg": float(np.nanmean([v for v in price_vars if not np.isnan(v)])) if price_vars else 0.0,
    }


# ----------------------------
# ANALYTICS (kept as-is, but seaborn is optional)
# ----------------------------
def compute_analytics_and_visuals(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comprehensive analytics CSVs and visualizations."""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping analytics visuals")
        return

    # Seaborn is optional; if missing, we fall back to matplotlib-only charts.
    try:
        import seaborn as sns  # type: ignore
        HAS_SEABORN = True
    except Exception:
        HAS_SEABORN = False
        sns = None  # type: ignore

    logger.info("Generating post-processing analytics and visuals...")

    # 1) Price Tier tagging
    df["price_tier"] = df["price_current"].map(tag_price_tier)

    # 2) Segment key (silhouette × material × price tier)
    df["segment_key"] = (
        df["silhouette"].fillna("other") + " | " +
        df["material"].fillna("unknown") + " | " +
        df["price_tier"]
    )

    # 3) SKU counts pivot (segment × retailer)
    pivot = (
        df.groupby(["segment_key", "source"])["product_id"]
        .count()
        .reset_index(name="n_skus")
        .pivot(index="segment_key", columns="source", values="n_skus")
        .fillna(0)
    )
    if "target" not in pivot.columns:
        pivot["target"] = 0
    if "gap" not in pivot.columns:
        pivot["gap"] = 0

    pivot["total"] = pivot.sum(axis=1)
    pivot["target_pct"] = pivot["target"] / pivot["total"] * 100
    pivot["gap_pct"] = pivot["gap"] / pivot["total"] * 100
    pivot.to_csv(output_dir / "segment_sku_counts.csv")
    logger.info("  → segment_sku_counts.csv")

    # 4) Price stats by segment
    price_stats = (
        df.groupby("segment_key")["price_current"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
    )
    price_stats.to_csv(output_dir / "segment_price_stats.csv", index=False)
    logger.info("  → segment_price_stats.csv")

    # 5) Premium participation
    premium = df[df["price_tier"] == "Premium"]
    overall = df["source"].value_counts()
    prem_counts = premium["source"].value_counts()
    premium_part = pd.DataFrame({
        "total_skus": overall,
        "premium_skus": prem_counts
    }).fillna(0)
    premium_part["premium_pct"] = premium_part["premium_skus"] / premium_part["total_skus"] * 100
    premium_part.to_csv(output_dir / "premium_participation.csv")
    logger.info("  → premium_participation.csv")

    # 6) Sale rates by segment and retailer
    df["is_sale_binary"] = df["is_sale"].map(lambda x: 1 if str(x).lower() in ("true", "1", "yes") else 0)
    sale_rates = (
        df.groupby(["segment_key", "source"])["is_sale_binary"]
        .mean()
        .reset_index(name="sale_rate")
    )
    sale_rates["sale_pct"] = sale_rates["sale_rate"] * 100
    sale_rates.to_csv(output_dir / "segment_sale_rates.csv", index=False)
    logger.info("  → segment_sale_rates.csv")

    # 7) Median price by segment × retailer
    med_price = (
        df.groupby(["segment_key", "source"])["price_current"]
        .median()
        .reset_index(name="median_price")
        .pivot(index="segment_key", columns="source", values="median_price")
        .fillna(0)
    )
    med_price.to_csv(output_dir / "segment_median_price.csv")
    logger.info("  → segment_median_price.csv")

    # 8) Opportunity score ranking
    opp = pivot.copy()
    med_map = med_price.to_dict(orient="index")
    opp["median_price_gap"] = opp.index.map(lambda k: med_map.get(k, {}).get("gap", 0))
    opp["opportunity_score"] = (
        opp["gap"] * opp["median_price_gap"] * (1 - opp["target_pct"] / 100)
    )
    opp = opp.sort_values("opportunity_score", ascending=False)
    opp.to_csv(output_dir / "segment_opportunity_ranked.csv")
    logger.info("  → segment_opportunity_ranked.csv")

    # === VISUALIZATIONS ===
    tier_order = ["Entry", "Mid", "Mid+", "Upper Mid", "Premium"]

    # Chart 1: SKU distribution by price tier (Target vs Gap)
    df_plot = (
        df.groupby(["price_tier", "source"])["product_id"]
        .count()
        .reset_index(name="count")
    )
    df_plot["price_tier"] = pd.Categorical(df_plot["price_tier"], categories=tier_order, ordered=True)
    df_plot = df_plot.sort_values("price_tier")

    fig, ax = plt.subplots(figsize=(10, 6))
    if HAS_SEABORN:
        sns.barplot(data=df_plot, x="price_tier", y="count", hue="source", ax=ax)
    else:
        # Matplotlib fallback: side-by-side bars
        xs = np.arange(len(tier_order))
        tgt = df_plot[df_plot["source"] == "target"].set_index("price_tier")["count"].reindex(tier_order).fillna(0).values
        gap = df_plot[df_plot["source"] == "gap"].set_index("price_tier")["count"].reindex(tier_order).fillna(0).values
        w = 0.35
        ax.bar(xs - w/2, tgt, width=w, label="target")
        ax.bar(xs + w/2, gap, width=w, label="gap")
        ax.set_xticks(xs)
        ax.set_xticklabels(tier_order)
    ax.set_title("SKU Distribution by Price Tier & Retailer")
    ax.set_xlabel("Price Tier")
    ax.set_ylabel("SKU Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sku_distribution_price.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → sku_distribution_price.png")

    # Chart 2: Retailer share heatmap (silhouette × price tier)
    heat = (
        df.groupby(["silhouette", "price_tier", "source"])["product_id"]
        .count()
        .reset_index(name="count")
        .pivot_table(index="silhouette", columns=["price_tier", "source"], values="count", fill_value=0)
    )
    heat_norm = heat.div(heat.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    if HAS_SEABORN:
        sns.heatmap(heat_norm, annot=True, fmt=".1f", ax=ax)
    else:
        ax.imshow(heat_norm.values, aspect="auto")
        ax.set_yticks(np.arange(heat_norm.shape[0]))
        ax.set_yticklabels(list(heat_norm.index))
        ax.set_xticks(np.arange(heat_norm.shape[1]))
        ax.set_xticklabels([f"{a}|{b}" for (a, b) in heat_norm.columns], rotation=90)
    ax.set_title("Retailer Share % by Silhouette × Price Tier")
    ax.set_xlabel("Price Tier × Retailer")
    ax.set_ylabel("Silhouette")
    plt.tight_layout()
    plt.savefig(output_dir / "retailer_share_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → retailer_share_heatmap.png")

    # Chart 3: Premium participation
    fig, ax = plt.subplots(figsize=(7, 5))
    premium_part["premium_pct"].plot(kind="bar", ax=ax)
    ax.set_title("Premium Tier Participation (%) by Retailer")
    ax.set_ylabel("% Premium SKUs")
    ax.set_xlabel("Retailer")
    ax.set_xticklabels(premium_part.index, rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "premium_participation.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → premium_participation.png")

    # Chart 4: Price spread boxplot
    df_box = df.copy()
    df_box["price_tier"] = pd.Categorical(df_box["price_tier"], categories=tier_order, ordered=True)
    df_box = df_box.sort_values("price_tier")

    fig, ax = plt.subplots(figsize=(12, 6))
    if HAS_SEABORN:
        sns.boxplot(data=df_box, x="price_tier", y="price_current", hue="source", ax=ax)
    else:
        # Matplotlib fallback: show only overall distribution by tier (no retailer split)
        data = [df_box[df_box["price_tier"] == t]["price_current"].values for t in tier_order]
        ax.boxplot(data, labels=tier_order, showfliers=False)
    ax.set_title("Price Distribution by Price Tier & Retailer")
    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(output_dir / "price_spread_boxplot.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → price_spread_boxplot.png")

    logger.info("Analytics generation complete!")


# ----------------------------
# VIZ helpers (no change to clustering)
# ----------------------------
def _project_2d(X: np.ndarray, method: str = "pca", random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    Project high-dim embeddings to 2D for visualization ONLY.
    method: pca | umap | tsne
    Returns: (X2, meta)
    """
    method = (method or "pca").lower()
    meta: dict = {"method": method}

    if method == "umap":
        if not HAS_UMAP:
            logger.warning("UMAP not installed; falling back to PCA for scatter.")
            method = "pca"
        else:
            reducer = umap.UMAP(
                n_neighbors=20,
                min_dist=0.12,
                metric="euclidean",
                random_state=random_state,
            )
            X2 = reducer.fit_transform(X)
            meta.update({"note": "UMAP preserves local neighborhoods better than PCA in 2D."})
            return X2, meta

    if method == "tsne":
        if not HAS_TSNE:
            logger.warning("t-SNE not available; falling back to PCA for scatter.")
            method = "pca"
        else:
            # t-SNE is slower; defaults tuned for ~few hundred points
            tsne = TSNE(
                n_components=2,
                perplexity=min(35, max(5, (X.shape[0] - 1) // 10)),
                learning_rate="auto",
                init="pca",
                random_state=random_state,
            )
            X2 = tsne.fit_transform(X)
            meta.update({"note": "t-SNE can show separation but distances are not globally meaningful."})
            return X2, meta

    # PCA default
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X)
    meta.update({
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
    })
    return X2, meta


def _plot_scatter_by_silhouette(
    df: pd.DataFrame,
    X2: np.ndarray,
    output_dir: Path,
    fname: str,
    title: str,
    hue_col: str = "silhouette",
    alpha: float = 0.65,
    s: int = 25,
    max_points: int | None = None,
) -> None:
    """Global scatter; optional downsample for readability."""
    if not HAS_MATPLOTLIB:
        return
    if max_points is not None and len(df) > max_points:
        # deterministic downsample
        idx = np.linspace(0, len(df) - 1, max_points).astype(int)
        d = df.iloc[idx].copy()
        Xp = X2[idx]
    else:
        d = df
        Xp = X2

    order = d[hue_col].value_counts().index.tolist()
    cmap = plt.cm.tab20
    colors = {k: cmap(i % 20) for i, k in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in order:
        m = (d[hue_col] == k).values
        ax.scatter(Xp[m, 0], Xp[m, 1], c=[colors[k]], label=str(k), alpha=alpha, s=s)
    ax.set_title(title)
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(output_dir / fname, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → %s", fname)


def _plot_per_silhouette_panels(
    df: pd.DataFrame,
    X2: np.ndarray,
    output_dir: Path,
    label_col: str = "subcluster",
    max_silhouettes: int = 12,
) -> None:
    """
    Creates one scatter per silhouette, colored by subcluster.
    This is the single most effective way to reduce clutter (same data, less overlap).
    """
    if not HAS_MATPLOTLIB:
        return

    sil_order = df["silhouette"].value_counts().index.tolist()[:max_silhouettes]
    cmap = plt.cm.tab20

    for sil in sil_order:
        sub = df[df["silhouette"] == sil].copy()
        if len(sub) < 5:
            continue
        idx = sub.index.values
        Xp = X2[idx]

        # cluster colors
        clus_order = sub[label_col].value_counts().index.tolist()
        colors = {k: cmap(i % 20) for i, k in enumerate(clus_order)}

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        for c in clus_order:
            m = (sub[label_col] == c).values
            ax.scatter(Xp[m, 0], Xp[m, 1], c=[colors[c]], label=str(c), alpha=0.65, s=28)
        ax.set_title(f"{sil}: subclusters in 2D (viz only)")
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        out = output_dir / f"taxonomy_{sil}_subclusters.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("  → %s", out.name)


def _plot_enhanced(
    df: pd.DataFrame,
    output_dir: Path,
    X_global: np.ndarray | None = None,
    viz_method: str = "pca",
    scatter_max_points: int = 450,
    do_per_silhouette: bool = True,
) -> None:
    """
    Generate distribution, price, retailer share, and scatter charts.
    IMPORTANT: clustering logic unchanged; scatter is visualization only.
    """
    if not HAS_MATPLOTLIB:
        return

    sil_order = df["silhouette"].value_counts().index.tolist()
    sil_counts = df.groupby("silhouette").agg(
        count=("product_id", "count"),
        target=("source", lambda s: (s == "target").sum()),
        gap=("source", lambda s: (s == "gap").sum()),
        median_price=("price_current", "median"),
    ).reindex(sil_order)
    x_pos = np.arange(len(sil_order))

    # 1) Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos, sil_counts["target"], label="Target", alpha=0.9)
    ax.bar(x_pos, sil_counts["gap"], bottom=sil_counts["target"], label="Gap", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Count")
    ax.set_title("Enhanced: Silhouette distribution by retailer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "taxonomy_enhanced_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → taxonomy_enhanced_distribution.png")

    # 2) Median price by silhouette (Target vs Gap)
    med_target = df[df["source"] == "target"].groupby("silhouette")["price_current"].median().reindex(sil_order)
    med_gap = df[df["source"] == "gap"].groupby("silhouette")["price_current"].median().reindex(sil_order)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width / 2, med_target.fillna(0), width, label="Target", alpha=0.9)
    ax.bar(x_pos + width / 2, med_gap.fillna(0), width, label="Gap", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Median price ($)")
    ax.set_title("Enhanced: Median price by silhouette (Target vs Gap)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "taxonomy_enhanced_price.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → taxonomy_enhanced_price.png")

    # 3) Retailer share
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - 0.175, 100 * sil_counts["target"] / sil_counts["count"], 0.35, label="Target %")
    ax.bar(x_pos + 0.175, 100 * sil_counts["gap"] / sil_counts["count"], 0.35, label="Gap %")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Share (%)")
    ax.set_title("Enhanced: Retailer share by silhouette")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "taxonomy_enhanced_retailer_share.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  → taxonomy_enhanced_retailer_share.png")

    # 4) Scatter: use PCA/UMAP/t-SNE
    if X_global is not None and X_global.shape[0] == len(df) and X_global.shape[1] >= 2:
        try:
            X2, meta = _project_2d(X_global, method=viz_method, random_state=42)

            # global silhouette scatter (downsample to reduce clutter)
            title = f"Enhanced: Silhouettes in 2D ({meta.get('method','pca').upper()} projection)"
            if meta.get("method") == "pca":
                title += f"\nPC1 {meta.get('pc1_var',0)*100:.1f}% | PC2 {meta.get('pc2_var',0)*100:.1f}%"
            _plot_scatter_by_silhouette(
                df=df,
                X2=X2,
                output_dir=output_dir,
                fname=f"taxonomy_enhanced_scatter_{meta.get('method','pca')}.png",
                title=title,
                hue_col="silhouette",
                alpha=0.60,
                s=24,
                max_points=scatter_max_points,
            )

            # per-silhouette subcluster plots (BEST for readability)
            if do_per_silhouette:
                _plot_per_silhouette_panels(df=df, X2=X2, output_dir=output_dir, label_col="subcluster")

        except Exception as ex:
            logger.warning("Scatter failed: %s", ex)


def run(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUT,
    use_tfidf_only: bool = False,
    min_silhouette_size: int = 10,
    min_cluster_size: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    doc_freq_threshold: float = 0.80,
    viz_method: str = "pca",
    scatter_max_points: int = 450,
    per_silhouette_plots: bool = True,
) -> None:
    logger.info("Loading %s ...", input_path)
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Schema compatibility: collapsed output uses price, is_on_sale; taxonomy expects price_current, is_sale
    if "price_current" not in df.columns and "price" in df.columns:
        df["price_current"] = df["price"]
    if "is_sale" not in df.columns and "is_on_sale" in df.columns:
        df["is_sale"] = df["is_on_sale"]
    if "product_id" not in df.columns and "product_url" in df.columns:
        df["product_id"] = df["product_url"].fillna("").astype(str)
    if "source" not in df.columns:
        df["source"] = "target"

    df["product_id"] = df["product_id"].astype(str)
    df["text_raw"] = df.apply(build_raw_text, axis=1)

    # 1) Brand tokens from CSV brand column
    brand_tokens = get_brand_tokens_from_column(df, brand_col="brand")
    if brand_tokens:
        logger.info("Brand tokens (from brand column): %s", sorted(list(brand_tokens))[:50])
    else:
        logger.info("Brand tokens: (none found / brand col missing)")

    # 2) Initial clean for statistical stopword detection
    base_stop = BASE_STOP | brand_tokens
    _temp = []
    for s in df["text_raw"]:
        t = _strip_promo(str(s) if pd.notna(s) else "")
        t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
        _temp.append(" ".join(w for w in t.split() if len(w) > 2 and w not in base_stop))
    statistical_stop = detect_statistical_stopwords(_temp, doc_freq_threshold=doc_freq_threshold)
    logger.info("Statistical stopwords (>%.0f%% doc freq): %s", doc_freq_threshold * 100, list(statistical_stop)[:20])

    # 3) Final cleaned text + derived fields
    df["text_clean"] = df["text_raw"].apply(lambda s: clean_text_enhanced(s, brand_tokens, statistical_stop))
    df["silhouette"] = df["text_raw"].map(classify_silhouette)
    df["material"] = df["text_raw"].map(extract_material)
    df["price_current"] = pd.to_numeric(df["price_current"], errors="coerce").fillna(0)

    logger.info("Silhouette distribution: %s", df["silhouette"].value_counts().to_dict())

    # Cluster fields
    df["subcluster"] = -1
    df["subcluster_score"] = np.nan
    df["segment_label"] = ""
    all_evals = []

    # --- CLUSTERING LOOP (UNCHANGED LOGIC) ---
    for sil in df["silhouette"].unique():
        block = df[df["silhouette"] == sil].copy()
        n = len(block)

        if n < min_silhouette_size:
            df.loc[block.index, "subcluster"] = 0
            df.loc[block.index, "segment_label"] = f"{sil} – small group – {_price_tier(block['price_current'].median())}"
            logger.info("  %s: %d items → single segment", sil, n)
            continue

        X = get_hybrid_embeddings(
            block,
            use_sentence_transformer=not use_tfidf_only,
            model_name=model_name,
            add_numeric=True,
        )
        labels = cluster_hdbscan(X, min_cluster_size=min_cluster_size)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        df.loc[block.index, "subcluster"] = labels
        block = block.copy()
        block["subcluster"] = labels

        ev = evaluate_clusters(block, X)
        ev["silhouette_name"] = sil
        ev["n"] = n
        ev["n_clusters"] = n_clusters
        all_evals.append(ev)
        df.loc[block.index, "subcluster_score"] = ev["silhouette"]

        mapping = label_cluster_enhanced(block)
        for c, lab in mapping.items():
            mask = (df["silhouette"] == sil) & (df["subcluster"] == c)
            df.loc[mask, "segment_label"] = f"{sil} – {lab}"

        logger.info("  %s: %d items → %d clusters, sil=%.3f, DB=%.2f",
                    sil, n, n_clusters, ev["silhouette"], ev["davies_bouldin"])

    # Global embeddings for scatter (viz only)
    X_global = get_hybrid_embeddings(
        df,
        use_sentence_transformer=not use_tfidf_only,
        model_name=model_name,
        add_numeric=True,
    )

    # Output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / "products_segmented_enhanced.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("  → %s", out_csv)

    # Post-processing analytics
    try:
        compute_analytics_and_visuals(df, output_dir)
    except Exception as ex:
        logger.warning("Analytics generation failed: %s", ex)

    # Visualizations (with UMAP/t-SNE options, plus per-silhouette plots)
    if HAS_MATPLOTLIB:
        _plot_enhanced(
            df=df,
            output_dir=output_dir,
            X_global=X_global,
            viz_method=viz_method,
            scatter_max_points=scatter_max_points,
            do_per_silhouette=per_silhouette_plots,
        )

    # Report
    ev_df = pd.DataFrame(all_evals)
    overall_sil = float(ev_df["silhouette"].mean()) if not ev_df.empty else 0.0
    overall_db = float(ev_df["davies_bouldin"].mean()) if not ev_df.empty else 0.0

    report_lines = [
        "# Enhanced Taxonomy Report",
        "",
        f"**Date:** {ts}",
        f"**Input:** {input_path.name}",
        "",
        "## Clustering quality",
        "",
        f"- **Mean silhouette (per silhouette):** {overall_sil:.4f}",
        f"- **Mean Davies-Bouldin:** {overall_db:.2f}",
        "",
        "| Silhouette | N | Clusters | Silhouette | Davies-Bouldin |",
        "|------------|---|----------|------------|----------------|",
    ]
    for _, r in ev_df.iterrows():
        report_lines.append(
            f"| {r['silhouette_name']} | {int(r['n'])} | {int(r['n_clusters'])} | {r['silhouette']:.3f} | {r['davies_bouldin']:.2f} |"
        )

    n_tgt = int((df["source"] == "target").sum()) if "source" in df.columns else 0
    n_gap = int((df["source"] == "gap").sum()) if "source" in df.columns else 0
    med_t = float(df[df["source"] == "target"]["price_current"].median()) if "source" in df.columns else 0.0
    med_g = float(df[df["source"] == "gap"]["price_current"].median()) if "source" in df.columns else 0.0

    report_lines.extend([
        "",
        "## Comparison (Target vs Gap)",
        "",
        "| Metric | Target | Gap |",
        "|--------|--------|-----|",
        f"| Count | {n_tgt} | {n_gap} |",
        f"| Median price | ${med_t:.1f} | ${med_g:.1f} |",
        "",
        "## Upgrades applied",
        "",
        "- Brand tokens from CSV brand column (source=retailer, brand=Champion/Kipling/etc.)",
        "- Statistical stopwords (configurable doc-freq threshold)",
        "- Hybrid embeddings: TF-IDF + bigrams + price bucket + material + is_sale",
        "- HDBSCAN within each silhouette (auto cluster count, noise → nearest cluster)",
        "- Structural word removal (zip, strap, pockets, closure)",
        "- NEW (viz only): PCA/UMAP/t-SNE 2D plots + per-silhouette subcluster plots (reduces clutter)",
        "",
        "## Segments (sample)",
        "",
    ])

    for sil in df["silhouette"].value_counts().index.tolist()[:5]:
        sub = df[df["silhouette"] == sil]
        for seg in sub["segment_label"].dropna().unique()[:3]:
            if not seg:
                continue
            cnt = int((df["segment_label"] == seg).sum())
            report_lines.append(f"- **{seg[:60]}** ({cnt} items)")

    if HAS_MATPLOTLIB:
        report_lines.extend([
            "",
            "## Visualizations",
            "",
            "![Distribution](taxonomy_enhanced_distribution.png)",
            "![Price](taxonomy_enhanced_price.png)",
            "![Retailer share](taxonomy_enhanced_retailer_share.png)",
            f"![Scatter](taxonomy_enhanced_scatter_{viz_method.lower()}.png)",
            "",
            "Per-silhouette plots are saved as:",
            "- `taxonomy_<silhouette>_subclusters.png` (one per silhouette)",
            "",
        ])

    out_report = output_dir / "taxonomy_enhanced_report.md"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info("  → %s", out_report)
    logger.info("Done. Mean silhouette=%.3f, mean DB=%.2f", overall_sil, overall_db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced taxonomy clustering")
    parser.add_argument("-n", "--normalized", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tfidf", action="store_true", help="TF-IDF only (no sentence-transformers)")
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--doc-freq", type=float, default=0.80)

    # NEW: viz-only controls (no impact on clustering)
    parser.add_argument("--viz", default="pca", choices=["pca", "umap", "tsne"], help="2D viz projection method (viz only)")
    parser.add_argument("--scatter-max", type=int, default=450, help="Max points in global scatter (downsample for readability)")
    parser.add_argument("--no-per-silhouette", action="store_true", help="Disable per-silhouette subcluster plots")

    args = parser.parse_args()

    run(
        input_path=args.normalized,
        output_dir=args.output_dir,
        use_tfidf_only=args.tfidf,
        min_silhouette_size=args.min_size,
        min_cluster_size=args.min_cluster_size,
        model_name=args.model,
        doc_freq_threshold=args.doc_freq,
        viz_method=args.viz,
        scatter_max_points=args.scatter_max,
        per_silhouette_plots=(not args.no_per_silhouette),
    )


if __name__ == "__main__":
    main()