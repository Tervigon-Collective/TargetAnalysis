#!/usr/bin/env python3
"""
Price coverage analysis: load from Chroma, clean/profile columns only, produce
two executive-ready analyses:
  1. Coverage gap in core professional ($60-$90) vs category leader
  2. Premium tier ($120+) under-participation vs 18% benchmark

Uses only: product_id, document, name, source (Gap|Target), price_current, price_original.
Best practice: column selection keeps processing fast; domain constants centralized.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Price coverage domain knowledge ──────────────────────────────────────────
COVERAGE_PRICE_BANDS = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 120), (120, 9999)]
COVERAGE_BAND_LABELS = ["0-20", "20-40", "40-60", "60-90", "90-120", "120+"]
# $60–90 = mid-tier sweet spot; work tote = tote + (work|laptop|professional|office)
PREMIUM_THRESHOLD = 120
PREMIUM_BENCHMARK_PCT = 18.0


def _safe_str(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _coerce_price(v: object) -> float:
    """Coerce to float; invalid/empty -> 0.0."""
    if v is None or v == "":
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        if not s:
            return 0.0
        return float(re.sub(r"[^\d.]", "", s))
    except (ValueError, TypeError):
        return 0.0


def _derive_source(product: dict) -> str:
    """Derive retailer source from product_url; fallback to brand."""
    url = _safe_str(product.get("product_url") or product.get("url") or "")
    url_lower = url.lower()
    if "gap.com" in url_lower:
        return "Gap"
    if "target.com" in url_lower:
        return "Target"
    brand = _safe_str(product.get("brand") or "").lower()
    if brand == "gap":
        return "Gap"
    if brand:
        return "Target"
    return "Unknown"


def load_from_chroma(
    collection_name: str = "target_handbags",
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
) -> list[dict]:
    """Load products from Chroma; return minimal records (6 columns only)."""
    import chromadb

    api_key = api_key or os.environ.get("CHROMA_API_KEY")
    if not api_key:
        raise ValueError("CHROMA_API_KEY required. Set in .env or pass --api-key.")

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

    records = []
    for i, pid in enumerate(ids):
        meta = (metadatas[i] if i < len(metadatas) else {}) or {}
        doc = documents[i] if i < len(documents) else ""
        p = dict(meta)
        p["product_id"] = p.get("product_id") or pid
        p["document"] = _safe_str(doc)
        p["name"] = _safe_str(p.get("name") or p.get("title") or "")
        p["source"] = _derive_source(p)
        p["price_current"] = meta.get("price_current")
        p["price_original"] = meta.get("price_original")
        records.append({
            "product_id": p["product_id"],
            "document": p["document"],
            "name": p["name"],
            "source": p["source"],
            "price_current": p["price_current"],
            "price_original": p["price_original"],
        })
    return records


def load_from_file(path: Path) -> list[dict]:
    """Load products from CSV/JSON/JSONL; return minimal records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("products", [data])
    elif suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    records = []
    for i, r in enumerate(rows):
        pid = _safe_str(r.get("product_id") or r.get("id") or r.get("tcin") or "")
        if not pid or pid.lower() in ("nan", "none"):
            url = _safe_str(r.get("product_url") or r.get("url") or "")
            match = re.search(r"preselect=(\d+)|/A-(\d+)|[-/](\d{8,})", url) if url else None
            pid = match.group(1) or match.group(2) or match.group(3) or f"row_{i}" if match else f"row_{i}"
        doc = _safe_str(r.get("document") or "")
        if not doc:
            doc = " ".join(filter(None, [
                _safe_str(r.get("name") or r.get("title")),
                _safe_str(r.get("brand")),
                _safe_str(r.get("description")),
            ]))
        name = _safe_str(r.get("name") or r.get("title") or "")
        p = dict(r)
        p["product_url"] = p.get("product_url") or p.get("url")
        source = _derive_source(p)
        records.append({
            "product_id": pid,
            "document": doc,
            "name": name,
            "source": source,
            "price_current": r.get("price_current") or r.get("price"),
            "price_original": r.get("price_original") or r.get("original_price"),
        })
    return records


def clean_and_profile(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    """
    Clean: coerce prices, drop invalid. Profile: counts, nulls, min/max.
    Returns (df, profile_dict).
    """
    df = pd.DataFrame(records)
    if df.empty:
        return df, {"raw_count": 0, "cleaned_count": 0}

    profile = {"raw_count": len(df)}
    price_curr = df["price_current"].apply(_coerce_price)
    price_orig = df["price_original"].apply(_coerce_price)
    df["price_numeric"] = np.where(price_curr > 0, price_curr, price_orig)
    df = df[df["price_numeric"] > 0].copy()
    profile["cleaned_count"] = len(df)
    profile["dropped"] = profile["raw_count"] - profile["cleaned_count"]
    if df.empty:
        return df, profile

    profile["price_min"] = float(df["price_numeric"].min())
    profile["price_max"] = float(df["price_numeric"].max())
    profile["source_dist"] = df["source"].value_counts().to_dict()
    profile["null_document"] = int(df["document"].isna().sum())
    profile["null_name"] = int(df["name"].isna().sum())
    return df, profile


def assign_price_band(price: float) -> str:
    for (lo, hi), label in zip(COVERAGE_PRICE_BANDS, COVERAGE_BAND_LABELS):
        if lo <= price < hi:
            return label
    return COVERAGE_BAND_LABELS[-1]


def assign_segment(text: str) -> str:
    """Work tote / tote / other based on document+name."""
    t = (text or "").lower()
    is_tote = "tote" in t
    is_work = bool(re.search(r"work|laptop|professional|office", t))
    if is_tote and is_work:
        return "work_tote"
    if is_tote:
        return "tote"
    return "other"


def compute_coverage_gap(df: pd.DataFrame, brand_compare: str = "Gap") -> tuple[str, list[str]]:
    """Coverage gap: band × segment × source. Returns (hero_line, report_lines)."""
    if "source" not in df.columns or "price_numeric" not in df.columns:
        return "", []
    df_cov = df.copy()
    df_cov["price_band"] = df_cov["price_numeric"].apply(assign_price_band)
    df_cov["_text"] = (df_cov["document"].fillna("") + " " + df_cov["name"].fillna("")).str.strip()
    df_cov["segment"] = df_cov["_text"].apply(assign_segment)

    source_vals = df_cov["source"].fillna("").astype(str).str.strip()
    is_compare = source_vals.str.lower() == brand_compare.lower()
    is_others = ~is_compare
    if is_compare.sum() == 0 or is_others.sum() == 0:
        return "", []

    df_cov["_group"] = np.where(is_compare, brand_compare, "Others")
    ct = df_cov.groupby(["price_band", "segment", "_group"]).size().unstack(fill_value=0)
    gap_col = brand_compare
    others_col = "Others"
    if gap_col not in ct.columns:
        gap_col = ct.columns[0]
    others_cols = [c for c in ct.columns if c != gap_col]
    others_col = others_cols[0] if others_cols else None
    if others_col is None:
        return "", []

    ct["gap_skus"] = ct[others_col] - ct[gap_col]
    gaps = ct[ct["gap_skus"] > 0].sort_values("gap_skus", ascending=False)

    hero = ""
    report_lines = []
    if len(gaps) > 0:
        top = gaps.iloc[0]
        idx = gaps.index[0]
        band, seg = idx[0], idx[1]
        n_comp = int(top[gap_col])
        n_oth = int(top[others_col])
        seg_label = "structured work totes" if seg == "work_tote" else ("totes" if seg == "tote" else seg)
        hero = (
            f"Coverage gap of {int(top['gap_skus'])} SKUs in ${band} {seg_label} "
            f"versus category leader ({n_oth} vs {n_comp})."
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
        for idx_row, row in gaps.head(10).iterrows():
            band, seg = idx_row[0], idx_row[1]
            seg_label = "work totes" if seg == "work_tote" else ("totes" if seg == "tote" else seg)
            report_lines.append(
                f"- ${band} {seg_label}: {int(row['gap_skus'])} SKU gap "
                f"({int(row[others_col])} Others vs {int(row[gap_col])} {brand_compare})"
            )
    else:
        ct["lead"] = ct[gap_col] - ct[others_col]
        close = ct[ct["lead"] >= 0].sort_values("lead", ascending=True)
        if len(close) > 0:
            top = close.iloc[0]
            idx = close.index[0]
            band, seg = idx[0], idx[1]
            seg_label = "work totes" if seg == "work_tote" else ("totes" if seg == "tote" else seg)
            hero = (
                f"No coverage gaps in current data. Closest segment: ${band} {seg_label} "
                f"({int(top[gap_col])} {brand_compare} vs {int(top[others_col])} Others)."
            )
        else:
            hero = "No coverage gaps identified with current data."
    return hero, report_lines


def create_price_band_chart(df: pd.DataFrame, brand_compare: str, out_path: Path) -> None:
    """Price band SKU distribution by source (Gap vs Others). Grouped bar chart."""
    import matplotlib.pyplot as plt

    df_plot = df.copy()
    df_plot["price_band"] = df_plot["price_numeric"].apply(assign_price_band)
    band_order = [b for b in COVERAGE_BAND_LABELS if b in df_plot["price_band"].unique()]
    df_plot["_group"] = np.where(
        df_plot["source"].str.lower() == brand_compare.lower(), brand_compare, "Others"
    )
    ct = df_plot.groupby(["price_band", "_group"]).size().unstack(fill_value=0)
    ct = ct.reindex(band_order).fillna(0)
    x = np.arange(len(ct.index))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if brand_compare in ct.columns:
        ax.bar(x - w / 2, ct[brand_compare], w, label=brand_compare, color="#2563eb")
    others_col = [c for c in ct.columns if c != brand_compare]
    if others_col:
        ax.bar(x + w / 2, ct[others_col[0]], w, label=others_col[0], color="#16a34a")
    ax.set_xlabel("Price band ($)")
    ax.set_ylabel("SKU count")
    ax.set_title("SKU density by price band — Gap vs Others")
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def create_premium_tier_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Premium tier (% SKUs above $120) per source vs 18% benchmark."""
    import matplotlib.pyplot as plt

    sources = sorted(df["source"].unique())
    pcts = []
    for src in sources:
        sub = df[df["source"] == src]
        n = len(sub)
        n_prem = (sub["price_numeric"] >= PREMIUM_THRESHOLD).sum()
        pcts.append(100 * n_prem / n if n else 0)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(sources))
    bars = ax.bar(x, pcts, color="#2563eb", alpha=0.8)
    ax.axhline(y=PREMIUM_BENCHMARK_PCT, color="#dc2626", linestyle="--", linewidth=2, label=f"Benchmark ({PREMIUM_BENCHMARK_PCT}%)")
    ax.set_xlabel("Source")
    ax.set_ylabel("% SKUs above $120")
    ax.set_title("Premium tier under-participation vs 18% benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, p) in enumerate(zip(bars, pcts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{p:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def compute_premium_tier(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Premium tier ($120+): % per source vs 18% benchmark."""
    if df.empty or "source" not in df.columns or "price_numeric" not in df.columns:
        return "", []
    report_lines = []
    total = len(df)
    premium = df[df["price_numeric"] >= PREMIUM_THRESHOLD]
    overall_pct = 100 * len(premium) / total if total else 0
    report_lines.append(
        f"Premium tier above ${PREMIUM_THRESHOLD} represents **{overall_pct:.1f}%** of SKUs "
        f"versus competitor average of {PREMIUM_BENCHMARK_PCT}%."
    )
    report_lines.extend([
        "",
        "**Why this hits hard:**",
        "- Signals margin mix weakness",
        "- Signals brand ceiling limitation",
        "- Signals lost AOV opportunity",
        "- Easy to quantify — moves conversation from volume to profitability",
        "",
        "**Per source:**",
        "",
    ])
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        n = len(sub)
        n_prem = (sub["price_numeric"] >= PREMIUM_THRESHOLD).sum()
        pct = 100 * n_prem / n if n else 0
        diff = pct - PREMIUM_BENCHMARK_PCT
        report_lines.append(f"- **{src}:** {pct:.1f}% premium ({n_prem}/{n} SKUs) — benchmark gap: {diff:+.1f}pp")
    hero = report_lines[0]
    return hero, report_lines


def _md_to_reportlab(line: str) -> str:
    """Convert markdown bold **x** to reportlab <b>x</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)


def export_to_pdf(
    report_text: str,
    chart_paths: list[Path],
    output_path: Path,
) -> None:
    """Export report text + chart images to a single PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

    doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    style_h1 = ParagraphStyle(name="CustomH1", parent=styles["Heading1"], fontSize=18, spaceAfter=12)
    style_h2 = ParagraphStyle(name="CustomH2", parent=styles["Heading2"], fontSize=14, spaceAfter=8)
    style_body = ParagraphStyle(name="CustomBody", parent=styles["Normal"], fontSize=10, spaceAfter=6)

    story = []
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_md_to_reportlab(line[2:]), style_h1))
        elif line.startswith("## "):
            story.append(Spacer(1, 12))
            story.append(Paragraph(_md_to_reportlab(line[3:]), style_h2))
        elif line.startswith("- "):
            story.append(Paragraph("&bull; " + _md_to_reportlab(line[2:]), style_body))
        else:
            story.append(Paragraph(_md_to_reportlab(line), style_body))

    for chart_path in chart_paths:
        if chart_path.exists():
            story.append(Spacer(1, 12))
            img = Image(str(chart_path), width=6 * inch, height=(6 * 5 / 8) * inch)
            story.append(img)

    doc.build(story)


def run_analysis(
    records: list[dict],
    output_dir: Path,
    brand_compare: str = "Gap",
    save_csv: bool = False,
    save_pdf: bool = False,
) -> None:
    """Clean, profile, analyze, write report and optional CSV."""
    df, profile = clean_and_profile(records)
    logger.info("Loaded %d records; after cleaning: %d", profile.get("raw_count", 0), profile.get("cleaned_count", 0))
    if profile.get("dropped"):
        logger.warning("Dropped %d rows with invalid/missing price", profile["dropped"])
    if df.empty:
        logger.error("No valid records to analyze.")
        return

    logger.info("Profile: price range $%.1f–$%.1f; sources: %s",
        profile.get("price_min", 0), profile.get("price_max", 0), profile.get("source_dist", {}))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"price_coverage_report_{ts}.md"
    csv_path = output_dir / f"price_coverage_cleaned_{ts}.csv" if save_csv else None
    chart1_path = output_dir / f"price_coverage_chart_bands_{ts}.png"
    chart2_path = output_dir / f"price_coverage_chart_premium_{ts}.png"
    pdf_path = output_dir / f"price_coverage_report_{ts}.pdf" if save_pdf else None

    report_lines = [
        "# Price Coverage Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Products: {profile.get('cleaned_count', 0)} (cleaned from {profile.get('raw_count', 0)})",
        "",
        "---",
        "",
        "## 1. Coverage gap in core professional price band",
        "",
    ]
    hero, cov_lines = compute_coverage_gap(df, brand_compare)
    report_lines.append(hero)
    report_lines.extend(cov_lines)

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Premium tier under-participation",
        "",
    ])
    prem_hero, prem_lines = compute_premium_tier(df)
    report_lines.append(prem_hero)
    report_lines.extend(prem_lines[1:])

    report_lines.extend([
        "",
        "### Charts",
        "",
        f"![Price band distribution]({chart1_path.name})",
        "",
        f"![Premium tier vs benchmark]({chart2_path.name})",
        "",
        "---",
        "",
        "## 3. Profile summary",
        "",
        f"- **Raw count:** {profile.get('raw_count', 0)}",
        f"- **Cleaned count:** {profile.get('cleaned_count', 0)}",
        f"- **Price range:** ${profile.get('price_min', 0):.1f} – ${profile.get('price_max', 0):.1f}",
        f"- **Source distribution:** {profile.get('source_dist', {})}",
        "",
    ])

    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Wrote %s", report_path)

    # Charts (always generated when PDF requested; also useful standalone)
    try:
        create_price_band_chart(df, brand_compare, chart1_path)
        logger.info("Wrote %s", chart1_path)
    except Exception as e:
        logger.warning("Price band chart failed: %s", e)
    try:
        create_premium_tier_chart(df, chart2_path)
        logger.info("Wrote %s", chart2_path)
    except Exception as e:
        logger.warning("Premium tier chart failed: %s", e)

    if save_pdf and pdf_path:
        try:
            chart_paths = [p for p in [chart1_path, chart2_path] if p.exists()]
            export_to_pdf(report_text, chart_paths, pdf_path)
            logger.info("Wrote %s", pdf_path)
        except Exception as e:
            logger.warning("PDF export failed: %s", e)

    if save_csv and csv_path:
        out_cols = ["product_id", "document", "name", "source", "price_current", "price_original", "price_numeric"]
        df[out_cols].to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price coverage analysis: coverage gap + premium tier vs 18%% benchmark."
    )
    parser.add_argument(
        "--from-chroma",
        action="store_true",
        help="Load from ChromaDB (default when no --input)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Load from CSV/JSON/JSONL instead of Chroma",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="target_handbags",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="Gap",
        help="Brand to compare vs Others (default: Gap)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Output directory",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save cleaned minimal CSV for audit",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Export report + charts to PDF",
    )
    args = parser.parse_args()

    if args.input:
        records = load_from_file(args.input)
        logger.info("Loaded %d records from %s", len(records), args.input)
    else:
        records = load_from_chroma(collection_name=args.collection)
        logger.info("Loaded %d records from Chroma collection '%s'", len(records), args.collection)

    run_analysis(
        records=records,
        output_dir=args.output_dir,
        brand_compare=args.brand,
        save_csv=args.save_csv,
        save_pdf=args.pdf,
    )


if __name__ == "__main__":
    main()
