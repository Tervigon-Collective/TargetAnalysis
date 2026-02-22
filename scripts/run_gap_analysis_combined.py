#!/usr/bin/env python3
"""
Combine Gap + Target handbags, then run gap analysis.
Usage:
  python scripts/run_gap_analysis_combined.py
  python scripts/run_gap_analysis_combined.py --gap FILE --target FILE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GAP = PROJECT_ROOT / "scraper" / "output" / "gap_products_export_e525a7cb.csv"
DEFAULT_TARGET = PROJECT_ROOT / "scraper" / "output" / "target_handbags_20260222_124633_normalized.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "combined_gap_target.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Gap + Target, run gap analysis")
    parser.add_argument("--gap", type=Path, default=DEFAULT_GAP, help="Gap products CSV")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target handbags CSV")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Combined output CSV")
    parser.add_argument("--no-run", action="store_true", help="Only combine, don't run analysis")
    args, rest = parser.parse_known_args()

    if not args.gap.exists():
        print(f"Error: Gap file not found: {args.gap}")
        sys.exit(1)
    if not args.target.exists():
        print(f"Error: Target file not found: {args.target}")
        sys.exit(1)

    g = pd.read_csv(args.gap)
    t = pd.read_csv(args.target)
    for c in g.columns:
        if c not in t.columns:
            t[c] = ""
    for c in t.columns:
        if c not in g.columns:
            g[c] = ""
    g = g[[c for c in t.columns]]
    combined = pd.concat([g, t], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Combined: {len(g)} Gap + {len(t)} Target = {len(combined)} rows -> {args.output}")

    if args.no_run:
        return

    sys.path.insert(0, str(SCRIPT_DIR))
    from run_gap_analysis import run_gap_analysis, load_products

    products = load_products(args.output)
    run_gap_analysis(
        output_dir=PROJECT_ROOT / "output",
        products=products,
        src_label=str(args.output),
        n_clusters=5,
        run_hdbscan=False,
        run_umap=False,
        use_metadata=True,
        balance_clustering=True,
        brand_compare="Gap",
    )


if __name__ == "__main__":
    main()
