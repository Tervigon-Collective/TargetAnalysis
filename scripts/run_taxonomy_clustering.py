#!/usr/bin/env python3
"""
Retail-ready taxonomy clustering: Silhouette → Material → Size/Feature.

Runs on products_normalized_combined.csv. Avoids brand/retailer copy dominating clusters by:
1. Rule-classifying into Silhouette (tote/crossbody/backpack/etc.)
2. Extracting attributes (material, closure, strap, size)
3. Embedding + clustering *within* each silhouette (best-k per group)
4. Auto-labeling: {Silhouette} – {Material/Feature} – {Price tier}

Usage:
    python scripts/run_taxonomy_clustering.py
    python scripts/run_taxonomy_clustering.py -n output/products_normalized_combined.csv --tfidf
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging = __import__("logging")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "output" / "products_normalized_combined.csv"
DEFAULT_OUT = PROJECT_ROOT / "output"

# --- Price coverage bands (aligned with run_price_coverage_analysis) ---
COVERAGE_BANDS = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 120), (120, 9999)]
COVERAGE_BAND_LABELS = ["0-20", "20-40", "40-60", "60-90", "90-120", "120+"]
PREMIUM_THRESHOLD = 120
PREMIUM_BENCHMARK_PCT = 18.0


def _assign_price_band(price: float) -> str:
    for (lo, hi), lbl in zip(COVERAGE_BANDS, COVERAGE_BAND_LABELS):
        if lo <= price < hi:
            return lbl
    return COVERAGE_BAND_LABELS[-1]


# --- 1) Text cleaning ---
GENERIC_STOP = frozenset({
    "the", "and", "for", "with", "this", "that", "your", "more", "made", "everything", "one",
    "women", "accessories", "accessory", "bag", "bags", "handbag", "handbags", "purse", "purses",
    "clothing", "shoes", "size", "our", "you", "from", "into", "also", "have", "can", "get",
})

BRANDS = frozenset({
    "vera", "bradley", "kipling", "champion", "narwey", "state", "shiraleah", "a new day",
    "target", "gap",
})

# Promo/spec patterns to strip before cleaning
_VIEWED_LIKE = re.compile(
    r"(?:Shipping\s*&\s*returns\s+)?(?:Customers\s+also\s+viewed|You\s+may\s+also\s+like).*",
    re.I | re.S,
)
_PROMO = [
    re.compile(r"\d+%\s*off\s*\+\s*extra\s+10%\s*in\s+the\s+app", re.I),
    re.compile(r"(?:extra\s+)?\d+%\s*off?\s*in\s+the\s+app", re.I),
    re.compile(r"\bextra\s+\d+%?\s*off\b", re.I),
    re.compile(r"\$\d+\.?\d*(?:\s+\$\d+\.?\d*)?", re.I),
    re.compile(r"\bSold\s*&\s*shipped\s+by\s+[^.]*\.?\s*", re.I),
    re.compile(r"\bAdd\s+to\s+Bag\b", re.I),
]
_SPEC = [
    (re.compile(r"\([HWDhwd]\)", re.I), " "),
    (re.compile(r"\b\d+\s*Inches?\s*\(\s*[HWD]\s*\)", re.I), " "),
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


def clean_text_for_taxonomy(s: str) -> str:
    """Remove brand/retailer/generic noise, keep silhouette/material/feature terms."""
    if pd.isna(s):
        return ""
    s = _strip_promo(str(s))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) > 2]
    toks = [t for t in toks if t not in GENERIC_STOP and t not in BRANDS]
    return " ".join(toks)


# --- 2) Silhouette classifier (priority order) ---
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


def classify_silhouette(raw: str) -> str:
    s = (raw or "").lower()
    for label, pat in SIL_RULES:
        if re.search(pat, s):
            return label
    return "other"


# --- 3) Attribute extraction ---
def _find(pattern: str, s: str) -> bool:
    return bool(re.search(pattern, s, re.I))


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


def extract_closure(s: str) -> str:
    s = (s or "").lower()
    if _find(r"\b(zip|zipper)\b", s):
        return "zip"
    if _find(r"\bmagnetic\b", s):
        return "magnetic"
    if _find(r"\bsnap\b", s):
        return "snap"
    if _find(r"\bbuckle\b", s):
        return "buckle"
    if _find(r"\bdrawstring\b", s):
        return "drawstring"
    return "unknown"


def extract_strap(s: str) -> str:
    s = (s or "").lower()
    if _find(r"\badjustable\b", s):
        return "adjustable"
    if _find(r"\bchain\b", s):
        return "chain"
    if _find(r"\bremovable\b", s):
        return "removable"
    return "unknown"


def extract_size(s: str) -> str:
    s = (s or "").lower()
    if _find(r"\bmini\b|\bmicro\b", s):
        return "mini"
    if _find(r"\bsmall\b", s):
        return "small"
    if _find(r"\blarge\b|\bspacious\b", s):
        return "large"
    return "unknown"


# --- 4) Embeddings ---
def get_embeddings(texts: list[str], use_tfidf: bool = False) -> np.ndarray:
    if use_tfidf:
        tfidf = TfidfVectorizer(max_features=2000, min_df=1, max_df=0.95, ngram_range=(1, 2))
        X = tfidf.fit_transform(texts)
        n_comp = min(64, X.shape[0] - 1, X.shape[1] - 1)
        n_comp = max(2, n_comp)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X = svd.fit_transform(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as ex:
        logger.warning("SentenceTransformer failed (%s), using TF-IDF", ex)
        return get_embeddings(texts, use_tfidf=True)


def best_kmeans(X: np.ndarray, k_min: int = 2, k_max: int = 12) -> dict | None:
    n = X.shape[0]
    k_max = min(k_max, n - 1, n // 2)
    if k_max < k_min:
        return None
    best = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            continue
        if best is None or score > best["score"]:
            best = {"k": k, "score": score, "labels": labels}
    return best


# --- 5) Auto-label ---
def _price_tier(med: float) -> str:
    if med < 25:
        return "Entry"
    if med < 60:
        return "Mid"
    if med < 120:
        return "Premium"
    return "Luxury"


def label_cluster(block: pd.DataFrame, subcluster_col: str = "subcluster") -> dict[int, str]:
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


def build_raw_text(row: dict) -> str:
    """Build raw text from normalized schema for classification & embedding."""
    def _s(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v).strip()
        return "" if s in ("", "nan", "None", "{}") else s

    parts = [
        _s(row.get("name")),
        _s(row.get("description")),
        _s(row.get("product_details")),
        _s(row.get("category_breadcrumb")),
        _s(row.get("material_text")) or _s(row.get("materials_section")),
        _s(row.get("bag_structure")),
        _s(row.get("interior_features")),
        _s(row.get("exterior_features")),
        _s(row.get("closure_type")),
        _s(row.get("handle_type")),
        _s(row.get("fabric_name")),
    ]
    return " ".join(p for p in parts if p)


def _build_comparison_section(df: pd.DataFrame) -> list[str]:
    """Build Target vs Gap comparison for all silhouettes and segments."""
    lines = ["", "## Comparison (Target vs Gap)", ""]

    # Overall
    n_tgt = (df["source"] == "target").sum()
    n_gap = (df["source"] == "gap").sum()
    n_total = len(df)
    med_tgt = df[df["source"] == "target"]["price_current"].median()
    med_gap = df[df["source"] == "gap"]["price_current"].median()
    sale_tgt = 100 * df[df["source"] == "target"]["is_sale"].map(
        lambda x: str(x).lower() in ("true", "1", "yes")
    ).sum() / n_tgt if n_tgt else 0
    sale_gap = 100 * df[df["source"] == "gap"]["is_sale"].map(
        lambda x: str(x).lower() in ("true", "1", "yes")
    ).sum() / n_gap if n_gap else 0

    lines.extend([
        "### Overall",
        "",
        f"| Metric | Target | Gap |",
        "|--------|--------|-----|",
        f"| Product count | {n_tgt} ({100*n_tgt/n_total:.1f}%) | {n_gap} ({100*n_gap/n_total:.1f}%) |",
        f"| Median price | ${med_tgt:.1f} | ${med_gap:.1f} |",
        f"| Sale % | {sale_tgt:.1f}% | {sale_gap:.1f}% |",
        "",
    ])

    # By silhouette
    lines.extend(["### By Silhouette", "", "| Silhouette | Target | Gap | Med Price (T) | Med Price (G) | Δ Price | Sale % (T) | Sale % (G) |", "|------------|--------|-----|---------------|---------------|---------|------------|------------|"])

    sil_order = df["silhouette"].value_counts().index.tolist()
    for sil in sil_order:
        sub = df[df["silhouette"] == sil]
        t = sub[sub["source"] == "target"]
        g = sub[sub["source"] == "gap"]
        t_cnt, g_cnt = len(t), len(g)
        med_t = t["price_current"].median() if len(t) else float("nan")
        med_g = g["price_current"].median() if len(g) else float("nan")
        delta = f"${med_t - med_g:+.1f}" if len(t) and len(g) and not (np.isnan(med_t) or np.isnan(med_g)) else "—"
        sale_t = 100 * t["is_sale"].map(lambda x: str(x).lower() in ("true", "1", "yes")).sum() / len(t) if len(t) else 0
        sale_g = 100 * g["is_sale"].map(lambda x: str(x).lower() in ("true", "1", "yes")).sum() / len(g) if len(g) else 0
        med_t_str = f"${med_t:.1f}" if len(t) and not np.isnan(med_t) else "—"
        med_g_str = f"${med_g:.1f}" if len(g) and not np.isnan(med_g) else "—"
        lines.append(f"| {sil} | {t_cnt} | {g_cnt} | {med_t_str} | {med_g_str} | {delta} | {sale_t:.0f}% | {sale_g:.0f}% |")

    # Competitive overlap
    lines.extend(["", "### Competitive Overlap", ""])
    segs_both = 0
    segs_tgt_only = 0
    segs_gap_only = 0
    for sil in df["silhouette"].unique():
        for seg in df[df["silhouette"] == sil]["segment_label"].unique():
            if not seg:
                continue
            r = df[(df["silhouette"] == sil) & (df["segment_label"] == seg)]
            t, g = (r["source"] == "target").sum(), (r["source"] == "gap").sum()
            if t > 0 and g > 0:
                segs_both += 1
            elif t > 0:
                segs_tgt_only += 1
            else:
                segs_gap_only += 1

    lines.extend([
        f"- **Segments with both retailers:** {segs_both}",
        f"- **Target-exclusive segments:** {segs_tgt_only}",
        f"- **Gap-exclusive segments:** {segs_gap_only}",
        "",
    ])
    return lines


def _build_conclusion_section(df: pd.DataFrame) -> list[str]:
    """Build conclusions and recommendations from the comparison."""
    lines = ["", "## Conclusion", ""]

    n_tgt = (df["source"] == "target").sum()
    n_gap = (df["source"] == "gap").sum()
    med_tgt = df[df["source"] == "target"]["price_current"].median()
    med_gap = df[df["source"] == "gap"]["price_current"].median()

    # Coverage
    tgt_sils = set(df[df["source"] == "target"]["silhouette"].unique())
    gap_sils = set(df[df["source"] == "gap"]["silhouette"].unique())
    tgt_only_sils = tgt_sils - gap_sils
    gap_only_sils = gap_sils - tgt_sils
    overlap_sils = tgt_sils & gap_sils

    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Dataset:** {len(df)} products ({n_tgt} Target, {n_gap} Gap)")
    lines.append(f"- **Price positioning:** Target median ${med_tgt:.1f} vs Gap median ${med_gap:.1f}" + (
        f" (Target {100*(med_tgt-med_gap)/med_gap:+.0f}% vs Gap)" if med_gap > 0 else ""
    ))
    lines.append(f"- **Shared silhouettes:** {len(overlap_sils)} — " + ", ".join(sorted(overlap_sils)))
    if tgt_only_sils:
        lines.append(f"- **Target-only silhouettes:** {', '.join(sorted(tgt_only_sils))}")
    if gap_only_sils:
        lines.append(f"- **Gap-only silhouettes:** {', '.join(sorted(gap_only_sils))}")
    lines.append("")

    lines.append("### Key Findings")
    lines.append("")
    findings = []
    if med_tgt < med_gap:
        findings.append("Target prices below Gap on median; entry/mid-tier positioning.")
    elif med_tgt > med_gap:
        findings.append("Target prices above Gap on median; premium or broader range.")
    if len(tgt_only_sils) > len(gap_only_sils):
        findings.append("Target has broader silhouette coverage.")
    elif len(gap_only_sils) > len(tgt_only_sils):
        findings.append("Gap has broader silhouette coverage in this dataset.")
    if gap_only_sils and "accessory" in gap_only_sils:
        findings.append("Gap leads in accessories/charms; Target opportunity to expand.")
    if tgt_only_sils and "backpack" in tgt_only_sils:
        findings.append("Target has backpack presence; Gap gap in that category.")
    if not findings:
        findings.append("Both retailers overlap across most silhouettes with similar price bands.")
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("### Recommendations")
    lines.append("")
    lines.append("- Use silhouette × segment view for assortment planning and competitor mapping.")
    lines.append("- Target-exclusive segments: consider Gap entry for white-space opportunity.")
    lines.append("- Gap-exclusive segments: consider Target entry for white-space opportunity.")
    lines.append("- Price-differential segments: benchmark and align if targeting same customer.")
    lines.append("")

    return lines


def _build_price_coverage_section(df: pd.DataFrame) -> list[str]:
    """Price coverage comparison: bands by silhouette, premium % vs benchmark."""
    df = df.copy()
    df["price_numeric"] = pd.to_numeric(df["price_current"], errors="coerce").fillna(0)
    df["price_band"] = df["price_numeric"].apply(_assign_price_band)

    lines = ["", "## Price coverage comparison", ""]
    lines.append("Price band distribution and premium tier ($120+) by silhouette, comparable to run_price_coverage_analysis.")
    lines.append("")

    sil_order = df["silhouette"].value_counts().index.tolist()
    band_order = [b for b in COVERAGE_BAND_LABELS if b in df["price_band"].unique()]

    # Table: silhouette | 0-20 | 20-40 | ... | 120+ | Premium %
    header = "| Silhouette | " + " | ".join(band_order) + f" | Premium % |"
    sep = "|" + "|".join(["---"] * (len(band_order) + 2)) + "|"
    lines.extend([header, sep])

    for sil in sil_order:
        sub = df[df["silhouette"] == sil]
        n_prem = (sub["price_numeric"] >= PREMIUM_THRESHOLD).sum()
        prem_pct = 100 * n_prem / len(sub) if len(sub) else 0
        row_vals = [str((sub["price_band"] == b).sum()) for b in band_order]
        lines.append(f"| {sil} | " + " | ".join(row_vals) + f" | {prem_pct:.1f}% |")

    total = len(df)
    overall_prem = (df["price_numeric"] >= PREMIUM_THRESHOLD).sum()
    lines.append("")
    lines.append(f"**Overall:** {100 * overall_prem / total:.1f}% premium (${PREMIUM_THRESHOLD}+) vs {PREMIUM_BENCHMARK_PCT}% benchmark.")
    lines.append("")
    return lines


def _plot_taxonomy(
    df: pd.DataFrame,
    output_dir: Path,
    ts: str,
    X_global: np.ndarray | None = None,
) -> list[Path]:
    """Generate taxonomy visualizations (silhouette-level + optional 2D scatter)."""
    saved: list[Path] = []
    if not HAS_MATPLOTLIB:
        return saved

    sil_order = df["silhouette"].value_counts().index.tolist()
    sil_counts = df.groupby("silhouette").agg(
        count=("product_id", "count"),
        target=("source", lambda s: (s == "target").sum()),
        gap=("source", lambda s: (s == "gap").sum()),
        median_price=("price_current", "median"),
    ).reindex(sil_order)

    n = len(sil_order)
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(n)]

    # 1) Silhouette distribution (stacked Target vs Gap)
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(n)
    ax.bar(x_pos, sil_counts["target"], label="Target", color="#CC0000", alpha=0.9)
    ax.bar(x_pos, sil_counts["gap"], bottom=sil_counts["target"], label="Gap", color="#0066CC", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Product count")
    ax.set_title("Taxonomy: Silhouette distribution by retailer (Target vs Gap)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p1 = output_dir / f"taxonomy_distribution_{ts}.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p1)

    # 2) Median price by silhouette and brand (Target vs Gap)
    med_target = df[df["source"] == "target"].groupby("silhouette")["price_current"].median().reindex(sil_order)
    med_gap = df[df["source"] == "gap"].groupby("silhouette")["price_current"].median().reindex(sil_order)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width / 2, med_target.fillna(0), width, label="Target", color="#CC0000", alpha=0.9)
    ax.bar(x_pos + width / 2, med_gap.fillna(0), width, label="Gap", color="#0066CC", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Median price ($)")
    ax.set_title("Taxonomy: Median price by silhouette (Target vs Gap)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p2 = output_dir / f"taxonomy_price_{ts}.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p2)

    # 3) Retailer share % by silhouette
    fig, ax = plt.subplots(figsize=(10, 5))
    tgt_pct = 100 * sil_counts["target"] / sil_counts["count"]
    gap_pct = 100 * sil_counts["gap"] / sil_counts["count"]
    width = 0.35
    ax.bar(x_pos - width / 2, tgt_pct, width, label="Target %", color="#CC0000", alpha=0.9)
    ax.bar(x_pos + width / 2, gap_pct, width, label="Gap %", color="#0066CC", alpha=0.9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Share (%)")
    ax.set_title("Taxonomy: Retailer share by silhouette (%)")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p3 = output_dir / f"taxonomy_retailer_share_{ts}.png"
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p3)

    # 4) 2D scatter (PCA of global embeddings, colored by silhouette)
    if X_global is not None and X_global.shape[0] == len(df) and X_global.shape[1] >= 2:
        try:
            n_comp = min(2, X_global.shape[1] - 1, X_global.shape[0] - 1)
            if n_comp >= 2:
                pca = PCA(n_components=2, random_state=42)
                X2 = pca.fit_transform(X_global)
                fig, ax = plt.subplots(figsize=(9, 7))
                for i, sil in enumerate(sil_order):
                    mask = df["silhouette"] == sil
                    ax.scatter(
                        X2[mask, 0],
                        X2[mask, 1],
                        c=[colors[i]],
                        label=sil,
                        alpha=0.6,
                        s=25,
                        edgecolors="white",
                        linewidth=0.3,
                    )
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
                ax.set_title("Taxonomy: Silhouettes in 2D (PCA projection)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout(rect=[0, 0, 0.85, 1])
                p4 = output_dir / f"taxonomy_scatter_{ts}.png"
                fig.savefig(p4, dpi=120, bbox_inches="tight")
                plt.close()
                saved.append(p4)
        except Exception as ex:
            logger.warning("Could not generate scatter: %s", ex)

    return saved


def _plot_taxonomy_price_coverage(df: pd.DataFrame, output_dir: Path, ts: str) -> list[Path]:
    """Price coverage charts: bands by silhouette, premium % vs benchmark."""
    saved: list[Path] = []
    if not HAS_MATPLOTLIB:
        return saved

    df = df.copy()
    df["price_numeric"] = pd.to_numeric(df["price_current"], errors="coerce").fillna(0)
    df["price_band"] = df["price_numeric"].apply(_assign_price_band)

    sil_order = df["silhouette"].value_counts().index.tolist()
    band_order = [b for b in COVERAGE_BAND_LABELS if b in df["price_band"].unique()]

    # 5) Price band distribution by silhouette (stacked bar)
    ct = df.groupby(["silhouette", "price_band"]).size().unstack(fill_value=0)
    ct = ct.reindex(index=sil_order, columns=band_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap_bands = plt.cm.viridis(np.linspace(0.2, 0.9, len(band_order)))
    bottom = np.zeros(len(sil_order))
    for j, band in enumerate(band_order):
        vals = ct[band].values if band in ct.columns else np.zeros(len(sil_order))
        ax.bar(
            np.arange(len(sil_order)),
            vals,
            bottom=bottom,
            label=band,
            color=cmap_bands[j],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(np.arange(len(sil_order)))
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("SKU count")
    ax.set_title("Taxonomy: Price band coverage by silhouette (stacked)")
    ax.legend(title="Price band ($)", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    p5 = output_dir / f"taxonomy_price_coverage_bands_{ts}.png"
    fig.savefig(p5, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p5)

    # 6) Premium tier % by silhouette vs 18% benchmark
    prem_pcts = []
    for sil in sil_order:
        sub = df[df["silhouette"] == sil]
        n = len(sub)
        n_prem = (sub["price_numeric"] >= PREMIUM_THRESHOLD).sum()
        prem_pcts.append(100 * n_prem / n if n else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#16a34a" if p >= PREMIUM_BENCHMARK_PCT else "#dc2626" for p in prem_pcts]
    bars = ax.bar(np.arange(len(sil_order)), prem_pcts, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(
        y=PREMIUM_BENCHMARK_PCT,
        color="#2563eb",
        linestyle="--",
        linewidth=2,
        label=f"Benchmark ({PREMIUM_BENCHMARK_PCT}%)",
    )
    ax.set_xticks(np.arange(len(sil_order)))
    ax.set_xticklabels(sil_order, rotation=45, ha="right")
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("% SKUs above $120")
    ax.set_title("Taxonomy: Premium tier ($120+) by silhouette vs benchmark")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, p in zip(bars, prem_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{p:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    p6 = output_dir / f"taxonomy_price_coverage_premium_{ts}.png"
    fig.savefig(p6, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(p6)

    return saved


def run(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUT,
    use_tfidf: bool = False,
    min_silhouette_size: int = 10,
) -> None:
    logger.info("Loading %s ...", input_path)
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["product_id"] = df["product_id"].astype(str)

    # Build raw text
    df["text_raw"] = df.apply(build_raw_text, axis=1)
    df["text_clean"] = df["text_raw"].map(clean_text_for_taxonomy)

    # Silhouette
    df["silhouette"] = df["text_raw"].map(classify_silhouette)

    # Attributes
    df["material"] = df["text_raw"].map(extract_material)
    df["closure"] = df["text_raw"].map(extract_closure)
    df["strap"] = df["text_raw"].map(extract_strap)
    df["size"] = df["text_raw"].map(extract_size)

    # Ensure price
    if "price_current" not in df.columns:
        df["price_current"] = 0.0
    df["price_current"] = pd.to_numeric(df["price_current"], errors="coerce").fillna(0)

    logger.info("Silhouette distribution: %s", df["silhouette"].value_counts().to_dict())

    # Per-silhouette clustering
    df["subcluster"] = -1
    df["subcluster_k"] = np.nan
    df["subcluster_score"] = np.nan
    df["segment_label"] = ""

    embed_method = "TF-IDF+SVD" if use_tfidf else "SentenceTransformer"
    logger.info("Using %s for embeddings", embed_method)

    for sil in df["silhouette"].unique():
        block = df[df["silhouette"] == sil].copy()
        n = len(block)

        if n < min_silhouette_size:
            # Small group: single segment
            df.loc[block.index, "subcluster"] = 0
            df.loc[block.index, "subcluster_k"] = 1
            df.loc[block.index, "subcluster_score"] = 0.0
            med = block["price_current"].median()
            df.loc[block.index, "segment_label"] = f"{sil} – small group – {_price_tier(float(med))}"
            logger.info("  %s: %d items → single segment", sil, n)
            continue

        texts = block["text_clean"].fillna("").tolist()
        texts = [t if t.strip() else "(no description)" for t in texts]
        X = get_embeddings(texts, use_tfidf)

        best = best_kmeans(X, k_min=2, k_max=min(12, n // 2))
        if best is None:
            df.loc[block.index, "subcluster"] = 0
            df.loc[block.index, "segment_label"] = f"{sil} – (no subclusters)"
            continue

        df.loc[block.index, "subcluster"] = best["labels"]
        df.loc[block.index, "subcluster_k"] = best["k"]
        df.loc[block.index, "subcluster_score"] = best["score"]

        block = block.copy()
        block["subcluster"] = best["labels"]
        mapping = label_cluster(block)
        for c, lab in mapping.items():
            mask = (df["silhouette"] == sil) & (df["subcluster"] == c)
            df.loc[mask, "segment_label"] = f"{sil} – {lab}"

        logger.info("  %s: %d items → k=%d, sil=%.3f", sil, n, best["k"], best["score"])

    # Global embeddings for scatter (same method as per-silhouette)
    all_texts = df["text_clean"].fillna("").tolist()
    all_texts = [t if t.strip() else "(no description)" for t in all_texts]
    X_global = get_embeddings(all_texts, use_tfidf)

    # Output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"products_segmented_taxonomy_{ts}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("  → %s", out_csv)

    # Summary report
    report_lines = [
        "# Taxonomy Segmentation Report",
        "",
        f"**Date:** {ts}",
        f"**Input:** {input_path.name}",
        "",
        "## Silhouette × Segment × Retailer",
        "",
        "| Silhouette | Segment | Count | Target | Gap | Med Price | Sale % |",
        "|------------|---------|-------|--------|-----|-----------|--------|",
    ]

    for sil in df["silhouette"].unique():
        sub = df[df["silhouette"] == sil]
        for seg in sub["segment_label"].unique():
            if not seg:
                continue
            row = df[(df["silhouette"] == sil) & (df["segment_label"] == seg)]
            n = len(row)
            tgt = (row["source"] == "target").sum()
            gap = (row["source"] == "gap").sum()
            med = row["price_current"].median()
            sale = 100 * row["is_sale"].map(lambda x: str(x).lower() in ("true", "1", "yes")).sum() / n if n else 0
            seg_safe = seg[:50].replace("|", "/")  # avoid breaking markdown table
            report_lines.append(f"| {sil} | {seg_safe} | {n} | {tgt} | {gap} | ${med:.1f} | {sale:.0f}% |")

    # Comparison section (Target vs Gap)
    report_lines.extend(_build_comparison_section(df))

    # Conclusion section
    report_lines.extend(_build_conclusion_section(df))

    # Visualizations
    if HAS_MATPLOTLIB:
        chart_files = _plot_taxonomy(df=df, output_dir=output_dir, ts=ts, X_global=X_global)
        price_cov_files = _plot_taxonomy_price_coverage(df=df, output_dir=output_dir, ts=ts)
        for p in chart_files + price_cov_files:
            logger.info("  → %s", p)
        report_lines.extend(_build_price_coverage_section(df))
        report_lines.extend(["", "## Visualizations", ""])
        for p in chart_files + price_cov_files:
            report_lines.append(f"![{p.name}]({p.name})")
            report_lines.append("")

    out_report = output_dir / f"taxonomy_report_{ts}.md"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info("  → %s", out_report)

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Taxonomy clustering: Silhouette → Material → Segment")
    parser.add_argument("-n", "--normalized", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tfidf", action="store_true", help="Use TF-IDF+SVD instead of sentence-transformers")
    parser.add_argument("--min-size", type=int, default=10, help="Min items to cluster within silhouette")
    args = parser.parse_args()

    run(
        input_path=args.normalized,
        output_dir=args.output_dir,
        use_tfidf=args.tfidf,
        min_silhouette_size=args.min_size,
    )


if __name__ == "__main__":
    main()
