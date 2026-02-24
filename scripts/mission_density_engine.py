#!/usr/bin/env python3
"""
Mission density computation and white space gap analysis.

Per spec: cells = Brand x Category x Mission x Price Band.
density = sku_count_in_cell / total_SKUs_in_category
competitor_median_density vs focus_brand_density; gap = competitor_median - focus_brand.

We always compare brands only. Target is a retailer; use brand (e.g. Universal Thread) for grouping.

Outputs: mission_density_cells.csv, white_space_gaps.csv

Usage:
    from mission_density_engine import compute_mission_density_and_gaps
    cells, gaps = compute_mission_density_and_gaps(rows, output_dir=Path("output"))

CLI (ChromaDB default, -i for CSV):
    python scripts/mission_density_engine.py
    python scripts/mission_density_engine.py -i output/products_mission_enriched.csv
    python scripts/mission_density_engine.py --focus-brand "universal thread" --competitors gap,jcrew
    python scripts/mission_density_engine.py --brands "universal thread",gap,jcrew -i output/products_mission_enriched.csv
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    np = None
    pd = None
    plt = None

try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
    sns = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "white_space.yaml"


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        import math
        if math.isnan(v):
            return ""
    s = str(v).strip()
    return "" if s in ("", "nan", "None", "{}") else s


def _leaf_from_breadcrumb(breadcrumb: str | None) -> str:
    """Derive leaf category from breadcrumb (e.g. 'Target > Clothing > Clutches' -> 'Clutches')."""
    if not breadcrumb or not str(breadcrumb).strip():
        return ""
    for sep in (">", "/"):
        parts = [p.strip() for p in str(breadcrumb).split(sep) if p.strip()]
        if parts:
            return parts[-1]
    return str(breadcrumb).strip()


def _load_white_space_config(config_path: Path | None = None) -> dict:
    import yaml
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {
            "white_space_price_bands": {
                "bounds": [0, 15, 25, 35, 50, 75, 9999],
                "labels": ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"],
            },
            "focus_brand": "",
        }
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _assign_price_band(price: float, config: dict) -> str:
    """Map price to white space price band label."""
    bands = config.get("white_space_price_bands") or {}
    bounds = bands.get("bounds") or [0, 15, 25, 35, 50, 75, 9999]
    labels = bands.get("labels") or ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
    p = float(price) if price is not None else 0
    for i, hi in enumerate(bounds[1:], start=0):
        lo = bounds[i]
        if lo <= p < hi:
            return labels[i] if i < len(labels) else labels[-1]
    return labels[-1] if labels else "75_plus"


def _get_primary_mission(row: dict) -> str:
    """Primary mission: mission_primary, or first of pipe-separated mission_tags."""
    prim = _s(row.get("mission_primary"))
    if prim:
        return prim
    tags = row.get("mission_tags")
    if isinstance(tags, str) and tags.strip():
        parts = [t.strip() for t in tags.split("|") if t.strip()]
        return parts[0] if parts else ""
    if isinstance(tags, (list, tuple)):
        return str(tags[0]).strip() if tags else ""
    return ""


def _get_category(row: dict) -> str:
    """Category: leaf_category or derive from category_breadcrumb."""
    leaf = _s(row.get("leaf_category"))
    if leaf:
        return leaf
    return _leaf_from_breadcrumb(row.get("category_breadcrumb") or row.get("category_breadcrumbs"))


def _get_brand_for_grouping(row: dict) -> str:
    """
    Brand for grouping and comparison. We always compare brands only.
    Target is a retailer; for Target products use brand (e.g. Universal Thread, A New Day).
    For Gap/J.Crew etc., use brand if present, else source (they are often both retailer and brand).
    Never use "target" (retailer) as brand; if source=target and brand empty, return "unknown".
    """
    brand = _s(row.get("brand") or "").strip().lower()
    source = _s(row.get("source") or "").lower()
    if brand and brand not in ("", "unknown"):
        return brand
    if source and source != "target":
        return source
    return "unknown"


def filter_rows_by_brands(
    rows: list[dict],
    brands_filter: list[str],
) -> list[dict]:
    """
    Filter rows to only include specified brands (grouping entities).
    Compares using _get_brand_for_grouping (brand names: Universal Thread, Gap, J.Crew, etc.).
    """
    filter_set = {b.lower().strip() for b in brands_filter}
    return [r for r in rows if _get_brand_for_grouping(r) in filter_set]


def _infer_focus_brand_from_target_rows(rows: list[dict]) -> str | None:
    """If focus is 'target' (retailer), infer focus_brand from most common brand in Target-sourced rows."""
    target_rows = [r for r in rows if _s(r.get("source")).lower() == "target"]
    if not target_rows:
        return None
    from collections import Counter
    brands = [_get_brand_for_grouping(r) for r in target_rows if _get_brand_for_grouping(r) not in ("unknown", "target")]
    if not brands:
        return None
    return Counter(brands).most_common(1)[0][0]


def compute_mission_density_and_gaps(
    rows: list[dict],
    output_dir: Path | None = None,
    config_path: Path | None = None,
    focus_brand: str | None = None,
    competitor_brands: list[str] | None = None,
    brands_filter: list[str] | None = None,
    target_brand: str | None = None,  # deprecated; use focus_brand
) -> tuple[list[dict], list[dict]]:
    """
    Compute mission density by cell and white space gaps (brand vs brand).

    We always compare brands only. Target is a retailer; use brand (e.g. Universal Thread) for grouping.

    Args:
        rows: Product rows with source, brand, leaf_category (or category_breadcrumb),
              mission_tags or mission_primary, price_current.
        output_dir: Directory to write mission_density_cells.csv, white_space_gaps.csv.
        config_path: Path to white_space.yaml.
        focus_brand: Brand to analyze for gaps (e.g. "universal thread", "gap"). Overrides config.
        competitor_brands: Other brands to compare (e.g. gap, jcrew). If None, all non-focus are competitors.
        brands_filter: If set, only include rows whose brand (grouping entity) is in this list.
        target_brand: Deprecated. Use focus_brand. If "target", infers focus from Target-sourced rows.

    Returns:
        (cells, gaps) - list of cell dicts, list of gap dicts.
    """
    config = _load_white_space_config(config_path)
    focus = (focus_brand or target_brand or config.get("focus_brand") or config.get("target_source", "")).lower().strip()
    comp_set = {b.lower().strip() for b in (competitor_brands or [])}
    filter_set = {b.lower().strip() for b in (brands_filter or [])}

    if filter_set:
        rows = filter_rows_by_brands(rows, list(filter_set))
        logger.info("Filtered to %d rows (brands: %s)", len(rows), sorted(filter_set))

    # Infer focus_brand when "target" (retailer) is specified
    if focus == "target":
        inferred = _infer_focus_brand_from_target_rows(rows)
        if inferred:
            focus = inferred
            logger.info("Inferred focus_brand from Target retailer rows: %s", focus)
        else:
            logger.warning("focus_brand='target' but no Target-sourced rows with brand; gap analysis may be empty.")

    # Assign per row: use brand for grouping (never target retailer as brand)
    for r in rows:
        r["_category"] = _get_category(r)
        r["_brand"] = _get_brand_for_grouping(r)
        r["_mission"] = _get_primary_mission(r)
        price = r.get("price_current") or r.get("price") or 0
        r["_price_band"] = _assign_price_band(price, config)

    valid = [r for r in rows if r["_category"] and r["_mission"]]
    if not valid:
        logger.warning("No rows with both category and mission; density outputs will be empty.")

    from collections import defaultdict
    cell_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    category_totals: dict[tuple[str, str], int] = defaultdict(int)

    for r in valid:
        key = (r["_brand"], r["_category"], r["_mission"], r["_price_band"])
        cell_counts[key] += 1
        cat_key = (r["_brand"], r["_category"])
        category_totals[cat_key] += 1

    # Cells: brand (grouping entity), category, mission, price_band
    cells: list[dict] = []
    for (brand, cat, mission, band), cnt in sorted(cell_counts.items()):
        cat_total = category_totals.get((brand, cat), 1)
        density = cnt / cat_total if cat_total else 0
        cells.append({
            "source": brand,  # keep "source" for backward compat; value is brand
            "brand": brand,
            "category": cat,
            "mission": mission,
            "price_band": band,
            "sku_count": cnt,
            "category_total": cat_total,
            "density": round(density, 6),
        })

    # Aggregate: focus_brand vs competitor brands
    comp_by_cell: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    focus_by_cell: dict[tuple[str, str, str], tuple[int, int, float]] = {}

    for c in cells:
        b, cat, mission, band = c["brand"], c["category"], c["mission"], c["price_band"]
        cell_key = (cat, mission, band)
        if b == focus:
            focus_by_cell[cell_key] = (c["sku_count"], c["category_total"], c["density"])
        elif not comp_set or b in comp_set:
            comp_by_cell[cell_key].append((c["density"], c["sku_count"]))

    import statistics
    gaps: list[dict] = []
    for cell_key, comp_list in comp_by_cell.items():
        cat, mission, band = cell_key
        focus_info = focus_by_cell.get(cell_key)
        focus_count = focus_info[0] if focus_info else 0
        focus_total = focus_info[1] if focus_info else 1
        focus_density = focus_info[2] if focus_info else 0.0

        comp_densities = [d for d, _ in comp_list]
        comp_counts = [cnt for _, cnt in comp_list]
        competitor_count = sum(comp_counts)
        competitor_median = statistics.median(comp_densities) if comp_densities else 0.0
        competitor_max = max(comp_densities) if comp_densities else 0.0
        gap = competitor_median - focus_density

        gaps.append({
            "category": cat,
            "mission": mission,
            "price_band": band,
            "focus_count_in_cell": focus_count,
            "competitor_count_in_cell": competitor_count,
            "focus_density": round(focus_density, 6),
            "competitor_median_density": round(competitor_median, 6),
            "competitor_max_density": round(competitor_max, 6),
            "gap": round(gap, 6),
        })

    # Keep legacy keys for backward compat
    for g in gaps:
        g["target_count_in_cell"] = g["focus_count_in_cell"]
        g["target_density"] = g["focus_density"]

    gaps.sort(key=lambda g: (-g["gap"], g["category"], g["mission"], g["price_band"]))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cells_path = output_dir / "mission_density_cells.csv"
        with open(cells_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "brand", "category", "mission", "price_band", "sku_count", "category_total", "density"])
            w.writeheader()
            w.writerows(cells)
        logger.info("Wrote %s", cells_path)

        gaps_path = output_dir / "white_space_gaps.csv"
        with open(gaps_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "category", "mission", "price_band",
                    "focus_count_in_cell", "competitor_count_in_cell",
                    "focus_density", "competitor_median_density", "competitor_max_density", "gap",
                    "target_count_in_cell", "target_density",
                ],
            )
            w.writeheader()
            w.writerows(gaps)
        logger.info("Wrote %s", gaps_path)

        if HAS_MATPLOTLIB and (cells or gaps):
            try:
                focus_label = focus or "focus brand"
                comp_label = ", ".join(sorted(comp_set)) if comp_set else "competitors"
                # When focus + competitors specified, show only those brands in charts (dynamic comparison)
                display_brands = None
                if focus and comp_set:
                    display_brands = [focus] + sorted(comp_set)
                plot_white_space_visuals(
                    cells, gaps, output_dir,
                    target_label=focus_label,
                    competitor_label=comp_label,
                    display_brands=display_brands,
                    focus_brand=focus,
                    competitor_brands=list(comp_set) if comp_set else None,
                )
            except Exception as ex:
                logger.warning("White space visualizations failed: %s", ex)

    return cells, gaps


def plot_white_space_visuals(
    cells: list[dict],
    gaps: list[dict],
    output_dir: Path,
    top_n_opportunities: int = 20,
    min_competitor_count: int = 2,
    target_label: str = "target",
    competitor_label: str = "competitors",
    display_brands: list[str] | None = None,
    focus_brand: str | None = None,
    competitor_brands: list[str] | None = None,
) -> None:
    """
    Generate charts and heatmaps for white space analysis and brand comparison.
    When display_brands is set (focus + competitors), charts show only those brands.
    Outputs: heatmaps, bar charts, and a summary report.
    """
    if not HAS_MATPLOTLIB or not gaps:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_cells = pd.DataFrame(cells) if cells else pd.DataFrame()
    # Filter to selected brands when doing a focused comparison (e.g. a new day vs abercrombie)
    if display_brands and not df_cells.empty:
        brand_set = {b.lower().strip() for b in display_brands}
        col = "brand" if "brand" in df_cells.columns else "source"
        df_cells = df_cells[df_cells[col].str.lower().str.strip().isin(brand_set)].copy()
    df_gaps = pd.DataFrame(gaps)

    # Filter gaps with meaningful competitor presence; fallback to 1 when no cells meet threshold
    df_gaps_sig = df_gaps[df_gaps["competitor_count_in_cell"] >= min_competitor_count].copy()
    if df_gaps_sig.empty and not df_gaps.empty:
        df_gaps_sig = df_gaps[df_gaps["competitor_count_in_cell"] >= 1].copy()

    # 1) Top white space opportunities - bar chart
    top = df_gaps_sig.head(top_n_opportunities).copy()
    if not top.empty:
        top["cell_label"] = top["category"].str[:25] + " | " + top["mission"] + " | " + top["price_band"]
        fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.35)))
        y_pos = np.arange(len(top))
        ax.barh(y_pos, top["gap"].values * 100, color="coral", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top["cell_label"].tolist(), fontsize=9)
        ax.set_xlabel("Gap (% density: Competitor Median - " + target_label + ")")
        ax.set_title("Top White Space Opportunities (" + target_label + " Under-Invested)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "white_space_top_opportunities.png", dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("  → white_space_top_opportunities.png")

    # 2) Gap by Mission × Price Band heatmap (distinct from top opportunities: aggregated view, no category)
    if not df_gaps.empty:
        pivot = df_gaps.pivot_table(
            index="mission", columns="price_band", values="gap", aggfunc="mean"
        ).fillna(0)
        band_order = ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
        pivot = pivot.reindex(columns=[c for c in band_order if c in pivot.columns], fill_value=0)
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, max(5, pivot.shape[0] * 0.4)))
            if HAS_SEABORN:
                sns.heatmap(pivot * 100, annot=True, fmt=".0f", ax=ax, cmap="YlOrRd", cbar_kws={"label": "Mean Gap (%)"})
            else:
                im = ax.imshow(pivot.values * 100, aspect="auto", cmap="YlOrRd")
                plt.colorbar(im, ax=ax, label="Mean Gap (%)")
                ax.set_yticks(np.arange(pivot.shape[0]))
                ax.set_yticklabels(pivot.index)
                ax.set_xticks(np.arange(pivot.shape[1]))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_title("Gap by Mission × Price Band (where opportunities concentrate)")
            plt.tight_layout()
            plt.savefig(output_dir / "white_space_gap_heatmap.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("  → white_space_gap_heatmap.png")

    # 2b) Gap by price band - mean gap per price band (which bands have biggest opportunity)
    if not df_gaps.empty:
        band_order = ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
        gap_by_band = df_gaps.groupby("price_band").agg({"gap": ["mean", "count"], "competitor_count_in_cell": "sum"}).reset_index()
        gap_by_band.columns = ["price_band", "mean_gap", "cell_count", "competitor_skus"]
        gap_by_band = gap_by_band[gap_by_band["price_band"].isin(band_order)]
        gap_by_band["price_band"] = pd.Categorical(gap_by_band["price_band"], categories=band_order, ordered=True)
        gap_by_band = gap_by_band.sort_values("price_band")
        if not gap_by_band.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(gap_by_band))
            bars = ax.bar(x, gap_by_band["mean_gap"].values * 100, color="coral", alpha=0.85, edgecolor="darkred")
            ax.set_xticks(x)
            ax.set_xticklabels(gap_by_band["price_band"], rotation=0)
            ax.set_xlabel("Price Band ($)")
            ax.set_ylabel("Mean Gap (%)")
            ax.set_title("White Space Gap by Price Band (higher = more opportunity)")
            for i, (g, c) in enumerate(zip(gap_by_band["mean_gap"] * 100, gap_by_band["cell_count"])):
                ax.text(i, g + 1, f"{g:.1f}%\n({int(c)} cells)", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / "white_space_gap_by_price_band.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("  → white_space_gap_by_price_band.png")

    # 3) Brand × Price Band - share of each brand's SKUs by price band
    if not df_cells.empty:
        col = "brand" if "brand" in df_cells.columns else "source"
        band_by_brand = df_cells.groupby([col, "price_band"])["sku_count"].sum().reset_index()
        brand_totals = df_cells.groupby(col)["sku_count"].sum()
        band_by_brand["pct"] = (band_by_brand["sku_count"] / band_by_brand[col].map(brand_totals).replace(0, 1)) * 100
        band_order = ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
        pivot_band = band_by_brand.pivot(index=col, columns="price_band", values="pct").fillna(0)
        pivot_band = pivot_band.reindex(columns=[c for c in band_order if c in pivot_band.columns], fill_value=0)
        if pivot_band.shape[0] > 0 and pivot_band.shape[1] > 0:
            fig, ax = plt.subplots(figsize=(10, max(5, pivot_band.shape[0] * 0.5)))
            pivot_band.plot(kind="barh", ax=ax, stacked=True, width=0.75)
            ax.set_xlabel("Share of Brand Assortment (%)")
            ax.set_ylabel("Brand")
            ax.set_title("Price Band Mix by Brand (% of each brand's SKUs per price band)")
            ax.legend(title="Price Band ($)", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / "white_space_price_band_by_brand.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("  → white_space_price_band_by_brand.png")

    # 3c) Category coverage by brand - top categories (SKU count)
    if not df_cells.empty:
        col = "brand" if "brand" in df_cells.columns else "source"
        cat_by_brand = df_cells.groupby([col, "category"])["sku_count"].sum().reset_index()
        top_cats = cat_by_brand.groupby("category")["sku_count"].sum().nlargest(12).index.tolist()
        df_cat = cat_by_brand[cat_by_brand["category"].isin(top_cats)]
        pivot_cat = df_cat.pivot(index="category", columns=col, values="sku_count").fillna(0)
        if pivot_cat.shape[0] > 0 and pivot_cat.shape[1] > 0:
            fig, ax = plt.subplots(figsize=(10, max(5, pivot_cat.shape[0] * 0.35)))
            pivot_cat.plot(kind="barh", ax=ax, stacked=False, width=0.8)
            ax.set_xlabel("SKU Count")
            ax.set_ylabel("Category")
            ax.set_title("Category Coverage by Brand (top categories by total SKUs)")
            ax.legend(title="Brand", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / "white_space_category_by_brand.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("  → white_space_category_by_brand.png")

    # 4) Mission share (%) by brand - bar comparison (avoids misleading raw SKU count)
    if not df_cells.empty:
        col = "brand" if "brand" in df_cells.columns else "source"
        sku_by_src_mission = df_cells.groupby([col, "mission"])["sku_count"].sum().reset_index()
        brand_totals = df_cells.groupby(col)["sku_count"].sum()
        sku_by_src_mission["brand_total"] = sku_by_src_mission[col].map(brand_totals)
        sku_by_src_mission["pct"] = (sku_by_src_mission["sku_count"] / sku_by_src_mission["brand_total"].replace(0, 1)) * 100
        pivot_pct = sku_by_src_mission.pivot(index="mission", columns=col, values="pct").fillna(0)
        if pivot_pct.shape[0] > 0 and pivot_pct.shape[1] > 0:
            pivot_pct = pivot_pct.loc[pivot_pct.sum(axis=1).sort_values(ascending=False).head(14).index]
            fig, ax = plt.subplots(figsize=(12, 7))
            pivot_pct.plot(kind="barh", ax=ax, stacked=False, width=0.8)
            ax.set_xlabel("Share of Brand Assortment (%)")
            ax.set_ylabel("Mission")
            ax.set_title("Mission Share (%) by Brand (% of each brand's SKUs in each mission)")
            ax.legend(title="Brand", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(output_dir / "white_space_sku_by_mission_brand.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("  → white_space_sku_by_mission_brand.png")

    # 5) Focus vs Competitor mission share (%) - grouped bar (avoids category mismatch from gap-based logic)
    # Compute mission-level % from cells so focus brand shows real share even when retailer category names differ
    if not df_cells.empty and focus_brand and competitor_brands:
        col = "brand" if "brand" in df_cells.columns else "source"
        sku_by_brand_mission = df_cells.groupby([col, "mission"])["sku_count"].sum().reset_index()
        brand_totals = df_cells.groupby(col)["sku_count"].sum()
        sku_by_brand_mission["pct"] = (
            sku_by_brand_mission["sku_count"] / sku_by_brand_mission[col].map(brand_totals).replace(0, 1)
        ) * 100
        focus_set = {focus_brand.lower().strip()}
        comp_set_vis = {b.lower().strip() for b in competitor_brands}
        focus_df = sku_by_brand_mission[sku_by_brand_mission[col].str.lower().str.strip().isin(focus_set)]
        comp_df = sku_by_brand_mission[sku_by_brand_mission[col].str.lower().str.strip().isin(comp_set_vis)]
        if not focus_df.empty and not comp_df.empty:
            all_missions = sorted(set(focus_df["mission"].unique()) | set(comp_df["mission"].unique()))
            focus_mission = focus_df.groupby("mission")["pct"].mean()
            comp_mission = comp_df.groupby("mission")["pct"].median()
            focus_vals = focus_mission.reindex(all_missions, fill_value=0).fillna(0)
            comp_vals = comp_mission.reindex(all_missions, fill_value=0).fillna(0)
            # Top 12 missions by competitor median
            top_missions = comp_vals.sort_values(ascending=False).head(12).index.tolist()
            if top_missions:
                t_vals = focus_vals.reindex(top_missions, fill_value=0).values
                c_vals = comp_vals.reindex(top_missions, fill_value=0).values
                t_plot = np.maximum(t_vals, 0.5)
                c_plot = np.maximum(c_vals, 0.5)
                x = np.arange(len(top_missions))
                w = 0.35
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.bar(x - w/2, t_plot, width=w, label=target_label + " (%)", color="steelblue", alpha=0.9)
                ax.bar(x + w/2, c_plot, width=w, label=competitor_label + " Median (%)", color="coral", alpha=0.85)
                for i, (tv, cv) in enumerate(zip(t_vals, c_vals)):
                    ax.text(x[i] - w/2, t_plot[i] + 0.5, f"{tv:.1f}%", ha="center", va="bottom", fontsize=8)
                    ax.text(x[i] + w/2, c_plot[i] + 0.5, f"{cv:.1f}%", ha="center", va="bottom", fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(top_missions, rotation=45, ha="right")
                ax.set_ylabel("Share of Brand Assortment (%)")
                ax.set_title(target_label + " vs " + competitor_label + " by Mission (Key Missions)")
                ax.legend()
                ax.set_ylim(0, max(c_plot.max(), t_plot.max()) * 1.2)
                plt.tight_layout()
                plt.savefig(output_dir / "white_space_target_vs_competitor.png", dpi=120, bbox_inches="tight")
                plt.close()
                logger.info("  → white_space_target_vs_competitor.png")
    # Fallback: use gap-based mission summary when focus/competitors not passed
    elif not df_gaps_sig.empty:
        mission_summary = df_gaps_sig.groupby("mission").agg({
            "target_density": "mean",
            "competitor_median_density": "mean",
            "competitor_count_in_cell": "sum",
        }).reset_index()
        mission_summary = mission_summary.nlargest(12, "competitor_count_in_cell")
        if not mission_summary.empty:
            x = np.arange(len(mission_summary))
            w = 0.35
            t_vals = (mission_summary["target_density"] * 100).values
            c_vals = (mission_summary["competitor_median_density"] * 100).values
            t_plot = np.maximum(t_vals, 0.5)
            c_plot = np.maximum(c_vals, 0.5)
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.bar(x - w/2, t_plot, width=w, label=target_label + " Avg", color="steelblue", alpha=0.9)
            ax.bar(x + w/2, c_plot, width=w, label=competitor_label + " Median", color="coral", alpha=0.85)
            for i, (tv, cv) in enumerate(zip(t_vals, c_vals)):
                ax.text(x[i] - w/2, t_plot[i] + 0.5, f"{tv:.1f}%", ha="center", va="bottom", fontsize=8)
                ax.text(x[i] + w/2, c_plot[i] + 0.5, f"{cv:.1f}%", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(mission_summary["mission"], rotation=45, ha="right")
            ax.set_ylabel("Density (%)")
            ax.set_title(target_label + " vs " + competitor_label + " Density by Mission (Key Missions)")
            ax.legend()
            ax.set_ylim(0, max(c_plot.max(), t_plot.max()) * 1.2)
            plt.tight_layout()
            plt.savefig(output_dir / "white_space_target_vs_competitor.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.info("  → white_space_target_vs_competitor.png")

    # 6) Summary report
    lines = [
        "# White Space Analysis Report",
        "",
        "## Summary",
        "",
        f"- **Focus brand:** {target_label}",
        f"- **Competitor brands:** {competitor_label}",
        f"- **Total gap cells:** {len(df_gaps)}",
        f"- **Gap cells with competitor_count >= {min_competitor_count}:** {len(df_gaps_sig)}",
        f"- **Top opportunity (gap):** {df_gaps_sig.iloc[0]['category']} | {df_gaps_sig.iloc[0]['mission']} | {df_gaps_sig.iloc[0]['price_band']} (gap={df_gaps_sig.iloc[0]['gap']*100:.1f}%)" if not df_gaps_sig.empty else "",
        "",
        "## Visualizations",
        "",
        f"- `white_space_top_opportunities.png` – Top cells by gap (specific category|mission|price cells)",
        "- `white_space_gap_heatmap.png` – Gap by mission × price band (aggregated, no category)",
        "- `white_space_gap_by_price_band.png` – Mean gap by price band (price lens only)",
        "- `white_space_price_band_by_brand.png` – Price band mix by brand (% of SKUs per band)",
        "- `white_space_category_by_brand.png` – Category coverage by brand (top categories)",
        "- `white_space_sku_by_mission_brand.png` – Mission share (%) by brand (% of each brand's SKUs per mission)",
        f"- `white_space_target_vs_competitor.png` – {target_label} vs {competitor_label} by mission",
        "",
        "## Understanding the Charts",
        "",
        f"**Why does {target_label} vs Competitors sometimes show {target_label} at 0%?**",
        f"When the focus brand has no (or few) products in mission cells where competitor brands invest, focus_density is 0%.",
        f"Those blue bars with \"0%\" labels indicate white space: competitors are present, focus brand is absent = opportunity.",
        "",
        "**Density** = % of that brand's category SKUs in that (mission, price_band) cell.",
        f"**Gap** = competitor_median_density - focus_density. Positive gap = {target_label} under-invested vs competitor brands.",
        "",
        "## Decision Guidance",
        "",
        "1. **High gap + high competitor_count** → Strong opportunity; consider sampling concepts.",
        "2. **High gap + low competitor_count** → Monitor; may be niche.",
        f"3. **Low gap ({target_label} ahead)** → Maintain or optimize; no expansion needed.",
    ]
    report_path = output_dir / "white_space_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("  → white_space_report.md")


def _load_from_chroma(collection_name: str, **chroma_kw) -> list[dict]:
    """Load products from ChromaDB (reuse tag_missions_and_cluster loader)."""
    import sys
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from tag_missions_and_cluster import load_from_chroma
    rows, _, _ = load_from_chroma(collection_name=collection_name, **chroma_kw)
    return rows


def _load_from_csv(path: Path) -> list[dict]:
    """Load products from CSV."""
    import csv
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Mission density and white space gaps. Load from ChromaDB by default."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="Input CSV (overrides ChromaDB; use only to load from file instead of DB).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "taxonomy_enhanced",
        help="Output directory (default: output/taxonomy_enhanced).",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("CHROMA_COLLECTION", "target_handbags"),
        help="ChromaDB collection (default: target_handbags).",
    )
    parser.add_argument(
        "--focus-brand",
        default=None,
        dest="focus_brand",
        help="Brand to analyze for white space gaps (e.g. 'universal thread', 'gap'). Comparisons are brand vs brand.",
    )
    parser.add_argument(
        "--target",
        default=None,
        dest="target_brand",
        help="(Deprecated) Use --focus-brand. If 'target', infers focus brand from Target-sourced rows.",
    )
    parser.add_argument(
        "--competitors",
        default=None,
        help="Comma-separated competitor brands/retailers (e.g. gap,jcrew,abercrombie). If omitted, all non-target are competitors.",
    )
    parser.add_argument(
        "--brands",
        default=None,
        help="Comma-separated brands to include (restricts analysis scope). If omitted, all brands in data are used.",
    )
    parser.add_argument("--chroma-api-key", default=os.environ.get("CHROMA_API_KEY"))
    parser.add_argument("--chroma-tenant", default=os.environ.get("CHROMA_TENANT"))
    parser.add_argument("--chroma-database", default=os.environ.get("CHROMA_DATABASE"))
    parser.add_argument("--chroma-url", default=os.environ.get("CHROMA_HTTP_URL"))
    parser.add_argument("--chroma-user", default=os.environ.get("CHROMA_HTTP_USER"))
    parser.add_argument("--chroma-password", default=os.environ.get("CHROMA_HTTP_PASSWORD"))
    args = parser.parse_args()

    if args.input is not None:
        logger.info("Loading from CSV: %s", args.input)
        rows = _load_from_csv(args.input)
    else:
        logger.info("Loading from ChromaDB (default): collection %s", args.collection)
        rows = _load_from_chroma(
            collection_name=args.collection,
            api_key=args.chroma_api_key,
            tenant=args.chroma_tenant,
            database=args.chroma_database,
            chroma_url=args.chroma_url,
            chroma_user=args.chroma_user,
            chroma_password=args.chroma_password,
        )
    logger.info("Loaded %d products.", len(rows))

    if not rows:
        logger.error("No products loaded.")
        raise SystemExit(1)

    focus_brand = (args.focus_brand or "").strip() or None
    target_brand = (args.target_brand or "").strip() or None
    competitor_brands = [b.strip() for b in args.competitors.split(",")] if args.competitors else None
    brands_filter = [b.strip() for b in args.brands.split(",")] if args.brands else None

    if focus_brand:
        logger.info("Focus brand: %s", focus_brand)
    elif target_brand:
        logger.info("Target/focus (may infer from data): %s", target_brand)
    if competitor_brands:
        logger.info("Competitor brands: %s", competitor_brands)
    if brands_filter:
        logger.info("Brands filter: %s", brands_filter)

    compute_mission_density_and_gaps(
        rows,
        output_dir=args.output_dir,
        focus_brand=focus_brand,
        target_brand=target_brand,
        competitor_brands=competitor_brands,
        brands_filter=brands_filter,
    )
    logger.info("Done.")
