#!/usr/bin/env python3
"""
Tag products with mission tags via LLM (Azure OpenAI) and cluster with embeddings.

Loads from ChromaDB (uses stored embeddings) or CSV (computes hybrid embeddings).
Outputs enriched CSV with mission_tags, segment_label, subcluster.

Usage:
    python scripts/tag_missions_and_cluster.py --from-chroma target_handbags -o output/products_mission_enriched.csv
    python scripts/tag_missions_and_cluster.py -i output/products_normalized_combined.csv -o output/products_mission_enriched.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent


def _s(v) -> str:
    if v is None or (isinstance(v, float) and (v != v or v == float("nan"))):
        return ""
    s = str(v).strip()
    return "" if s in ("", "nan", "None", "{}") else s


def _get_chroma_client(
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    chroma_url: str | None = None,
    chroma_user: str | None = None,
    chroma_password: str | None = None,
):
    """Create ChromaDB client (Cloud or self-hosted)."""
    import chromadb
    from chromadb.config import Settings

    api_key = api_key or os.environ.get("CHROMA_API_KEY")
    if api_key:
        kwargs: dict = {"api_key": api_key}
        if tenant or os.environ.get("CHROMA_TENANT"):
            kwargs["tenant"] = tenant or os.environ.get("CHROMA_TENANT")
        if database or os.environ.get("CHROMA_DATABASE"):
            kwargs["database"] = database or os.environ.get("CHROMA_DATABASE")
        return chromadb.CloudClient(**kwargs)

    url = chroma_url or os.environ.get("CHROMA_HTTP_URL")
    if not url and os.environ.get("CHROMA_HTTP_HOST"):
        host = os.environ.get("CHROMA_HTTP_HOST", "localhost")
        port = os.environ.get("CHROMA_HTTP_PORT", "8000")
        url = f"http://{host}:{port}"
    url = url or "http://localhost:8000"
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    settings_kw: dict = {}
    user = chroma_user or os.environ.get("CHROMA_HTTP_USER")
    pwd = chroma_password or os.environ.get("CHROMA_HTTP_PASSWORD")
    if not user and not pwd:
        creds = os.environ.get("CHROMA_AUTH_CREDENTIALS", "")
        if ":" in creds:
            user, _, pwd = creds.partition(":")
    if user and pwd:
        settings_kw = {
            "chroma_client_auth_provider": "chromadb.auth.basic_authn.BasicAuthClientProvider",
            "chroma_client_auth_credentials": f"{user}:{pwd}",
        }
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(**settings_kw) if settings_kw else Settings(),
    )


def load_from_chroma(
    collection_name: str,
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    chroma_url: str | None = None,
    chroma_user: str | None = None,
    chroma_password: str | None = None,
) -> tuple[list[dict], list[list[float]] | None]:
    """
    Load products from ChromaDB with metadata, documents, and embeddings.

    Returns (rows, embeddings). embeddings may be None if collection has no embeddings.
    """
    client = _get_chroma_client(
        api_key=api_key,
        tenant=tenant,
        database=database,
        chroma_url=chroma_url,
        chroma_user=chroma_user,
        chroma_password=chroma_password,
    )
    collection = client.get_collection(name=collection_name)
    result = collection.get(include=["metadatas", "documents", "embeddings"])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or [""] * len(ids)
    embeddings = result.get("embeddings")

    rows = []
    for i, pid in enumerate(ids):
        meta = (metadatas[i] if i < len(metadatas) else {}) or {}
        meta["product_id"] = meta.get("product_id") or pid
        meta["document"] = documents[i] if i < len(documents) else ""
        meta["name"] = _s(meta.get("name") or meta.get("title"))
        rows.append(meta)

    emb_list = None
    if embeddings is not None and len(embeddings) == len(rows):
        emb_list = embeddings
    return rows, emb_list


def load_from_csv(path: Path) -> list[dict]:
    """Load products from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["product_id"] = _s(r.get("product_id") or r.get("tcin") or r.get("id", ""))
        r["name"] = _s(r.get("name") or r.get("product_name") or r.get("title"))
    return rows


def get_hybrid_embeddings_for_df(df: pd.DataFrame) -> np.ndarray:
    """Compute hybrid embeddings for CSV-sourced data. Reuses taxonomy logic."""
    import sys
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from run_taxonomy_clustering_enhanced import (
        build_raw_text,
        get_hybrid_embeddings,
        classify_silhouette,
        extract_material,
        get_brand_tokens_from_column,
        detect_statistical_stopwords,
        clean_text_enhanced,
        _strip_promo,
        BASE_STOP,
    )
    import re

    df = df.copy()
    df["text_raw"] = df.apply(lambda r: build_raw_text(r.to_dict()), axis=1)
    brand_tokens = get_brand_tokens_from_column(df, brand_col="brand")
    base_stop = BASE_STOP | brand_tokens
    _temp = []
    for s in df["text_raw"]:
        t = _strip_promo(str(s) if pd.notna(s) else "")
        t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
        _temp.append(" ".join(w for w in t.split() if len(w) > 2 and w not in base_stop))
    statistical_stop = detect_statistical_stopwords(_temp, doc_freq_threshold=0.80)
    df["text_clean"] = df["text_raw"].apply(
        lambda s: clean_text_enhanced(s, brand_tokens, statistical_stop)
    )
    if "price_current" not in df.columns and "price" in df.columns:
        df["price_current"] = df["price"]
    if "is_sale" not in df.columns and "is_on_sale" in df.columns:
        df["is_sale"] = df["is_on_sale"]
    df["material"] = df["text_raw"].map(extract_material)
    df["price_current"] = pd.to_numeric(df["price_current"], errors="coerce").fillna(0)
    X = get_hybrid_embeddings(
        df,
        text_col="text_clean",
        use_sentence_transformer=True,
        add_numeric=True,
    )
    return X


def cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """HDBSCAN clustering; noise (-1) reassigned to nearest non-noise neighbor."""
    try:
        import hdbscan
    except ImportError:
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
        _, nearest = nn.kneighbors(X[noise_mask])
        labels[noise_mask] = labels[valid_idx[nearest.ravel()]]
    return labels


def compute_mission_penetration(rows: list[dict], mission_col: str = "mission_tags") -> list[dict]:
    """Per-mission SKU counts and pct of total."""
    total = len(rows)
    from collections import Counter
    counts: Counter = Counter()
    for r in rows:
        tags = r.get(mission_col) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split("|") if t.strip()]
        for t in tags:
            counts[t] += 1
    return [
        {"mission_id": m, "sku_count": c, "pct_of_total": round(100 * c / total, 1) if total else 0}
        for m, c in counts.most_common()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag products with mission tags (LLM) and cluster with embeddings."
    )
    parser.add_argument(
        "--from-chroma",
        metavar="COLLECTION",
        help="Load from ChromaDB collection (uses stored embeddings).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input CSV (fallback when not using --from-chroma).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "products_mission_enriched.csv",
        help="Output CSV path (default: output/products_mission_enriched.csv).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM tagging (for testing clustering only; mission_tags will be empty).",
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip clustering (only tag missions).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Products per LLM request (default: 5).",
    )
    # ChromaDB args (for --from-chroma)
    parser.add_argument("--chroma-api-key", default=os.environ.get("CHROMA_API_KEY"))
    parser.add_argument("--chroma-tenant", default=os.environ.get("CHROMA_TENANT"))
    parser.add_argument("--chroma-database", default=os.environ.get("CHROMA_DATABASE"))
    parser.add_argument(
        "--chroma-url",
        default=os.environ.get("CHROMA_HTTP_URL") or "http://localhost:8000",
    )
    parser.add_argument("--chroma-user", default=os.environ.get("CHROMA_HTTP_USER"))
    parser.add_argument("--chroma-password", default=os.environ.get("CHROMA_HTTP_PASSWORD"))
    args = parser.parse_args()

    if not args.from_chroma and not args.input:
        default_csv = PROJECT_ROOT / "output" / "products_normalized_combined.csv"
        if default_csv.exists():
            args.input = default_csv
            logger.info("Using default input: %s", args.input)
        else:
            parser.error("Provide --from-chroma COLLECTION or -i INPUT.csv")

    # Load data
    rows: list[dict] = []
    embeddings: list[list[float]] | np.ndarray | None = None
    from_chroma = bool(args.from_chroma)

    if args.from_chroma:
        logger.info("Loading from ChromaDB collection '%s'...", args.from_chroma)
        rows, embeddings = load_from_chroma(
            collection_name=args.from_chroma,
            api_key=args.chroma_api_key,
            tenant=args.chroma_tenant,
            database=args.chroma_database,
            chroma_url=args.chroma_url,
            chroma_user=args.chroma_user,
            chroma_password=args.chroma_password,
        )
        logger.info("Loaded %d products from ChromaDB.", len(rows))
    else:
        rows = load_from_csv(args.input)
        logger.info("Loaded %d products from CSV.", len(rows))

    if not rows:
        logger.error("No products loaded.")
        return

    # Mission tagging
    if not args.skip_llm:
        import sys
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        from llm_mission_tagger import tag_missions_batch
        logger.info("Tagging missions via LLM (batch_size=%d)...", args.batch_size)
        tag_lists = tag_missions_batch(rows, batch_size=args.batch_size)
        for i, tags in enumerate(tag_lists):
            rows[i]["mission_tags"] = "|".join(tags) if tags else ""
    else:
        for r in rows:
            r["mission_tags"] = ""

    # Clustering
    if not args.skip_cluster:
        if from_chroma and embeddings is not None:
            X = np.array(embeddings, dtype=np.float32)
            logger.info("Using ChromaDB embeddings (shape=%s)", X.shape)
        else:
            df = pd.DataFrame(rows)
            if "price_current" not in df.columns:
                df["price_current"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0)
            if "source" not in df.columns:
                df["source"] = "target"
            logger.info("Computing hybrid embeddings for clustering...")
            X = get_hybrid_embeddings_for_df(df)
        labels = cluster_hdbscan(X, min_cluster_size=args.min_cluster_size)
        for i, lab in enumerate(labels):
            rows[i]["subcluster"] = int(lab)
        n_clusters = len(set(labels))
        logger.info("Clustering done: %d clusters.", n_clusters)
    else:
        for r in rows:
            r["subcluster"] = -1

    # Mission penetration
    penetration = compute_mission_penetration(rows)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    penetration_path = out_dir / "mission_penetration.csv"
    with open(penetration_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mission_id", "sku_count", "pct_of_total"])
        w.writeheader()
        w.writerows(penetration)
    logger.info("Wrote %s", penetration_path)

    # Output CSV
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out_row = {k: ("" if v is None else str(v)) for k, v in r.items()}
            w.writerow(out_row)
    logger.info("Wrote %s", args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
