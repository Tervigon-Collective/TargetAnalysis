#!/usr/bin/env python3
"""Validate white space graphs against underlying CSV data."""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "universal_thread_vs_all"


def load_data():
    cells = pd.read_csv(OUTPUT_DIR / "mission_density_cells.csv")
    gaps = pd.read_csv(OUTPUT_DIR / "white_space_gaps.csv")
    return cells, gaps


def validate_top_opportunities(cells, gaps):
    """1. white_space_top_opportunities.png - Top cells by gap."""
    min_competitor_count = 2
    top_n = 20
    df_gaps_sig = gaps[gaps["competitor_count_in_cell"] >= min_competitor_count]
    if df_gaps_sig.empty:
        df_gaps_sig = gaps[gaps["competitor_count_in_cell"] >= 1]
    top = df_gaps_sig.head(top_n)
    expected_labels = (top["category"].str[:25] + " | " + top["mission"] + " | " + top["price_band"]).tolist()
    expected_gaps = (top["gap"] * 100).values
    return {
        "expected_top_20_labels": expected_labels,
        "expected_top_20_gaps_pct": expected_gaps.tolist(),
        "first_opportunity": f"{top.iloc[0]['category']} | {top.iloc[0]['mission']} | {top.iloc[0]['price_band']} = {top.iloc[0]['gap']*100:.1f}%",
    }


def validate_gap_heatmap(gaps):
    """2. white_space_gap_heatmap.png - mean gap by mission x price_band."""
    pivot = gaps.pivot_table(index="mission", columns="price_band", values="gap", aggfunc="mean").fillna(0)
    band_order = ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
    pivot = pivot.reindex(columns=[c for c in band_order if c in pivot.columns], fill_value=0)
    return {"pivot_shape": pivot.shape, "sample_values": pivot.iloc[:3, :3].to_dict()}


def validate_gap_by_price_band(gaps):
    """2b. white_space_gap_by_price_band.png - mean gap per price band."""
    band_order = ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
    gap_by_band = gaps.groupby("price_band").agg(
        {"gap": ["mean", "count"], "competitor_count_in_cell": "sum"}
    ).reset_index()
    gap_by_band.columns = ["price_band", "mean_gap", "cell_count", "competitor_skus"]
    gap_by_band = gap_by_band[gap_by_band["price_band"].isin(band_order)]
    gap_by_band["price_band"] = pd.Categorical(gap_by_band["price_band"], categories=band_order, ordered=True)
    gap_by_band = gap_by_band.sort_values("price_band")
    return gap_by_band.to_dict(orient="records")


def validate_price_band_by_brand(cells):
    """3. white_space_price_band_by_brand.png - share of each brand's SKUs by price band."""
    col = "brand" if "brand" in cells.columns else "source"
    band_by_brand = cells.groupby([col, "price_band"])["sku_count"].sum().reset_index()
    brand_totals = cells.groupby(col)["sku_count"].sum()
    band_by_brand["pct"] = (band_by_brand["sku_count"] / band_by_brand[col].map(brand_totals).replace(0, 1)) * 100
    band_order = ["0_15", "15_25", "25_35", "35_50", "50_75", "75_plus"]
    pivot = band_by_brand.pivot(index=col, columns="price_band", values="pct").fillna(0)
    pivot = pivot.reindex(columns=[c for c in band_order if c in pivot.columns], fill_value=0)
    # Sum per brand should be 100
    row_sums = pivot.sum(axis=1)
    return {
        "brands": pivot.index.tolist(),
        "brand_row_sums_pct": {b: round(s, 2) for b, s in row_sums.items()},
        "universal_thread_pct_by_band": pivot.loc["universal thread"].to_dict() if "universal thread" in pivot.index else {},
    }


def validate_category_by_brand(cells):
    """3c. white_space_category_by_brand.png - top categories by brand."""
    col = "brand" if "brand" in cells.columns else "source"
    cat_by_brand = cells.groupby([col, "category"])["sku_count"].sum().reset_index()
    top_cats = cat_by_brand.groupby("category")["sku_count"].sum().nlargest(12).index.tolist()
    df_cat = cat_by_brand[cat_by_brand["category"].isin(top_cats)]
    pivot = df_cat.pivot(index="category", columns=col, values="sku_count").fillna(0)
    # Spot check: universal thread Shoulder Bags
    ut_shoulder = cells[(cells["brand"] == "universal thread") & (cells["category"] == "Shoulder Bags")]["sku_count"].sum()
    return {
        "top_12_categories": top_cats,
        "universal_thread_shoulder_bags_sku_count": int(ut_shoulder),
        "pivot_shape": pivot.shape,
    }


def validate_sku_by_mission_brand(cells):
    """4. white_space_sku_by_mission_brand.png - mission share % by brand."""
    col = "brand" if "brand" in cells.columns else "source"
    sku_by_src_mission = cells.groupby([col, "mission"])["sku_count"].sum().reset_index()
    brand_totals = cells.groupby(col)["sku_count"].sum()
    sku_by_src_mission["brand_total"] = sku_by_src_mission[col].map(brand_totals)
    sku_by_src_mission["pct"] = (
        sku_by_src_mission["sku_count"] / sku_by_src_mission["brand_total"].replace(0, 1)
    ) * 100
    pivot = sku_by_src_mission.pivot(index="mission", columns=col, values="pct").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(14).index]
    # Sum per brand across missions should be 100
    col_sums = pivot.sum(axis=0)
    ut_total = cells[cells["brand"] == "universal thread"]["sku_count"].sum()
    ut_mission_sum = cells[cells["brand"] == "universal thread"].groupby("mission")["sku_count"].sum()
    return {
        "brand_col_sums_pct": {b: round(s, 2) for b, s in col_sums.items()},
        "universal_thread_total_skus": int(ut_total),
        "universal_thread_mission_breakdown": ut_mission_sum.to_dict(),
    }


def validate_mission_sku_heatmap(cells):
    """4b. white_space_mission_sku_count_heatmap.png - density % by brand, labels = SKU count."""
    col = "brand" if "brand" in cells.columns else "source"
    sku_by_src_mission = cells.groupby([col, "mission"])["sku_count"].sum().reset_index()
    brand_totals = cells.groupby(col)["sku_count"].sum()
    sku_by_src_mission["brand_total"] = sku_by_src_mission[col].map(brand_totals)
    sku_by_src_mission["pct"] = (
        sku_by_src_mission["sku_count"] / sku_by_src_mission["brand_total"].replace(0, 1)
    ) * 100
    pivot_sku = sku_by_src_mission.pivot(index=col, columns="mission", values="sku_count").fillna(0)
    pivot_pct = sku_by_src_mission.pivot(index=col, columns="mission", values="pct").fillna(0)
    top_missions = pivot_sku.sum().sort_values(ascending=False).head(14).index.tolist()
    pivot_sku = pivot_sku[[c for c in top_missions if c in pivot_sku.columns]]
    pivot_pct = pivot_pct[[c for c in top_missions if c in pivot_pct.columns]].reindex_like(pivot_sku).fillna(0)
    # universal thread row
    ut_row_sku = pivot_sku.loc["universal thread"] if "universal thread" in pivot_sku.index else pd.Series()
    ut_row_pct = pivot_pct.loc["universal thread"] if "universal thread" in pivot_pct.index else pd.Series()
    return {
        "universal_thread_sku_by_mission": ut_row_sku.to_dict(),
        "universal_thread_pct_by_mission": {k: round(v, 4) for k, v in ut_row_pct.to_dict().items()},
    }


def validate_target_vs_competitor(cells, focus_brand="universal thread"):
    """5. white_space_target_vs_competitor.png - focus vs competitors by mission."""
    col = "brand" if "brand" in cells.columns else "source"
    brands = cells[col].unique().tolist()
    comp_brands = [b for b in brands if str(b).lower().strip() != focus_brand.lower()]
    sku_by_brand_mission = cells.groupby([col, "mission"])["sku_count"].sum().reset_index()
    brand_totals = cells.groupby(col)["sku_count"].sum()
    sku_by_brand_mission["pct"] = (
        sku_by_brand_mission["sku_count"] / sku_by_brand_mission[col].map(brand_totals).replace(0, 1)
    ) * 100
    focus_df = sku_by_brand_mission[sku_by_brand_mission[col].str.lower().str.strip() == focus_brand.lower()]
    comp_df = sku_by_brand_mission[sku_by_brand_mission[col].str.lower().str.strip().isin([b.lower() for b in comp_brands])]
    all_missions = sorted(set(focus_df["mission"].unique()) | set(comp_df["mission"].unique()))
    focus_mission = focus_df.groupby("mission")["pct"].mean()
    comp_mission = comp_df.groupby("mission")["pct"].median()
    focus_vals = focus_mission.reindex(all_missions, fill_value=0).fillna(0)
    comp_vals = comp_mission.reindex(all_missions, fill_value=0).fillna(0)
    top_missions = comp_vals.sort_values(ascending=False).head(12).index.tolist()
    t_vals = focus_vals.reindex(top_missions, fill_value=0).values
    c_vals = comp_vals.reindex(top_missions, fill_value=0).values
    return {
        "top_12_missions": top_missions,
        "focus_pct": {m: round(t_vals[i], 2) for i, m in enumerate(top_missions)},
        "competitor_median_pct": {m: round(c_vals[i], 2) for i, m in enumerate(top_missions)},
    }


def run_validation():
    cells, gaps = load_data()
    results = {}
    errors = []

    # 1. Top opportunities
    r1 = validate_top_opportunities(cells, gaps)
    results["top_opportunities"] = r1
    report_top = f"{r1['first_opportunity']}"
    if "Awayday tote" not in report_top and "gap=100" not in report_top:
        # Report says Awayday tote | minimal_modern | 75_plus (gap=100.0%)
        expected_top = "Awayday tote | minimal_modern | 75_plus"
        if expected_top not in str(r1["expected_top_20_labels"][:3]):
            errors.append(f"Top opportunity mismatch: expected '{expected_top}' in top 3, got {r1['expected_top_20_labels'][:3]}")
    else:
        pass  # OK

    # 2. Gap heatmap
    r2 = validate_gap_heatmap(gaps)
    results["gap_heatmap"] = r2

    # 2b. Gap by price band
    r2b = validate_gap_by_price_band(gaps)
    results["gap_by_price_band"] = r2b
    band_sums = sum(x["mean_gap"] * x["cell_count"] for x in r2b)
    manual_mean = gaps["gap"].mean() * 100
    results["gap_by_price_band_weighted_check"] = f"manual mean gap % = {manual_mean:.2f}"

    # 3. Price band by brand
    r3 = validate_price_band_by_brand(cells)
    results["price_band_by_brand"] = r3
    for b, s in r3["brand_row_sums_pct"].items():
        if abs(s - 100) > 1:
            errors.append(f"Price band by brand: {b} row sum = {s}% (expected ~100%)")

    # 4. Category by brand
    r4 = validate_category_by_brand(cells)
    results["category_by_brand"] = r4

    # 5. SKU by mission brand
    r5 = validate_sku_by_mission_brand(cells)
    results["sku_by_mission_brand"] = r5
    for b, s in r5["brand_col_sums_pct"].items():
        if abs(s - 100) > 1:
            errors.append(f"Mission by brand: {b} col sum = {s}% (expected ~100%)")

    # 6. Mission heatmap
    r6 = validate_mission_sku_heatmap(cells)
    results["mission_sku_heatmap"] = r6

    # 7. Target vs competitor
    r7 = validate_target_vs_competitor(cells)
    results["target_vs_competitor"] = r7

    return results, errors


def main():
    print("=" * 60)
    print("White Space Graph Validation: output/universal_thread_vs_all")
    print("=" * 60)
    cells, gaps = load_data()
    print(f"\nData loaded: {len(cells)} cells, {len(gaps)} gap rows")
    print(f"Brands in cells: {sorted(cells['brand'].unique().tolist())}")

    results, errors = run_validation()

    print("\n--- VALIDATION SUMMARY ---\n")
    for name, data in results.items():
        print(f"[{name}]")
        if isinstance(data, dict):
            for k, v in list(data.items())[:5]:
                print(f"  {k}: {v}")
        else:
            print(f"  {data}")
        print()

    if errors:
        print("--- ISSUES FOUND ---")
        for e in errors:
            print(f"  ! {e}")
    else:
        print("--- All validations passed (no data inconsistencies). ---")

    # Cross-check with report
    report_path = OUTPUT_DIR / "white_space_report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        if "Top opportunity" in report:
            from re import search
            m = search(r"Top opportunity \(gap\): (.+) \(gap=([\d.]+)%\)", report)
            if m:
                rep_cat_mission_band, rep_gap = m.group(1), float(m.group(2))
                top_gap = results["top_opportunities"]["expected_top_20_gaps_pct"][0]
                top_label = results["top_opportunities"]["expected_top_20_labels"][0]
                if abs(top_gap - rep_gap) > 0.1:
                    print(f"\n  Report top opportunity gap mismatch: report={rep_gap}%, computed={top_gap}%")
                else:
                    print(f"\n  Report matches data: top opportunity = {top_label} ({top_gap:.1f}%)")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
