#!/usr/bin/env python3
"""
Combine retailers or load from ChromaDB, run taxonomy enrichment, then gap analysis.

Default: Load all data from ChromaDB, run taxonomy, then gap analysis.

CSV mode: Combine Gap + Target handbags from CSVs, then run analysis.
  python scripts/run_gap_analysis_combined.py --gap FILE --target FILE

ChromaDB mode (default):
  python scripts/run_gap_analysis_combined.py
  python scripts/run_gap_analysis_combined.py --collection target_handbags
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DEFAULT_GAP = PROJECT_ROOT / "scraper" / "output" / "gap_products_export_e525a7cb.csv"
DEFAULT_TARGET = PROJECT_ROOT / "scraper" / "output" / "target_handbags_20260222_124633_normalized.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "combined_gap_target.csv"
TAXONOMY_OUTPUT_DIR = PROJECT_ROOT / "output" / "taxonomy_gap_combined"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap analysis: load from ChromaDB (default) or combine CSVs, run taxonomy, then gap analysis."
    )
    parser.add_argument("--gap", type=Path, default=None, help="Gap products CSV (enables CSV mode with --target)")
    parser.add_argument("--target", type=Path, default=None, help="Target handbags CSV (enables CSV mode with --gap)")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Combined output CSV (CSV mode only)")
    parser.add_argument("--no-run", action="store_true", help="Only combine/save, don't run analysis")
    parser.add_argument("--skip-taxonomy", action="store_true", help="Skip taxonomy step; use raw combined data")
    parser.add_argument(
        "--collection",
        type=str,
        default="target_handbags",
        help="ChromaDB collection name (default mode)",
    )
    parser.add_argument(
        "--brands",
        type=str,
        default="auto",
        help='Comma-separated brands to compare, or "auto" for all. Example: "Gap,Universal Thread,A New Day"',
    )
    args = parser.parse_args()

    use_csv_mode = args.gap is not None or args.target is not None
    if use_csv_mode and (args.gap is None or args.target is None):
        print("Error: Both --gap and --target are required for CSV mode.")
        sys.exit(1)

    products = None
    src_label = "ChromaDB"

    if use_csv_mode:
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
        src_label = str(args.output)

        if not args.no_run and not args.skip_taxonomy:
            sys.path.insert(0, str(SCRIPT_DIR))
            from run_taxonomy_clustering_enhanced import run as run_taxonomy

            run_taxonomy(
                input_path=args.output,
                output_dir=TAXONOMY_OUTPUT_DIR,
            )
            taxonomy_csv = TAXONOMY_OUTPUT_DIR / "products_segmented_enhanced.csv"
            if taxonomy_csv.exists():
                products = pd.read_csv(taxonomy_csv).to_dict("records")
                src_label = str(taxonomy_csv)
                print(f"Loaded taxonomy-enriched: {len(products)} products")
        elif not args.no_run:
            products = pd.read_csv(args.output).to_dict("records")
    else:
        if args.no_run:
            print("Error: --no-run only applies to CSV mode.")
            sys.exit(1)
        sys.path.insert(0, str(SCRIPT_DIR))
        from run_taxonomy_clustering_enhanced import run as run_taxonomy

        run_taxonomy(
            input_path=None,
            output_dir=TAXONOMY_OUTPUT_DIR,
            collection_name=args.collection,
            chroma_api_key=os.environ.get("CHROMA_API_KEY"),
            chroma_tenant=os.environ.get("CHROMA_TENANT"),
            chroma_database=os.environ.get("CHROMA_DATABASE"),
            chroma_url=os.environ.get("CHROMA_HTTP_URL"),
            chroma_user=os.environ.get("CHROMA_HTTP_USER"),
            chroma_password=os.environ.get("CHROMA_HTTP_PASSWORD"),
        )
        taxonomy_csv = TAXONOMY_OUTPUT_DIR / "products_segmented_enhanced.csv"
        if taxonomy_csv.exists():
            products = pd.read_csv(taxonomy_csv).to_dict("records")
            src_label = f"ChromaDB ({args.collection}) + taxonomy"
            print(f"Loaded from ChromaDB + taxonomy: {len(products)} products")
        else:
            print("Error: Taxonomy did not produce products_segmented_enhanced.csv")
            sys.exit(1)

    if args.no_run:
        return

    if products is None:
        print("Error: No products to analyze.")
        sys.exit(1)

    from run_gap_analysis import run_gap_analysis

    brands_arg = args.brands.strip().lower() if args.brands else ""
    if brands_arg == "" or brands_arg == "none":
        brands_param: list[str] | None = []
    elif brands_arg == "auto":
        brands_param = None
    else:
        brands_param = [b.strip() for b in args.brands.split(",") if b.strip()]

    run_gap_analysis(
        output_dir=PROJECT_ROOT / "output",
        products=products,
        src_label=src_label,
        n_clusters=5,
        run_hdbscan=False,
        run_umap=False,
        use_metadata=True,
        balance_clustering=True,
        brands=brands_param,
    )


if __name__ == "__main__":
    main()
