#!/usr/bin/env python3
"""
Tag products with mission tags via LLM (Azure OpenAI) and cluster with embeddings.

Always loads from ChromaDB by default (uses stored embeddings, fetches whole dataset).
Skips LLM for products that already have mission_tags in DB; persists mission_tags
and cluster labels back to ChromaDB after tagging/clustering. Use -i CSV to load from
file instead (no DB persistence).

Usage:
    python scripts/tag_missions_and_cluster.py -o output/products_mission_enriched.csv
    python scripts/tag_missions_and_cluster.py --collection target_handbags -o output/products_mission_enriched.csv
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


def _leaf_from_breadcrumb(breadcrumb: str | None) -> str:
    """Derive leaf category from breadcrumb (e.g. 'Target > Clothing > Clutches' -> 'Clutches')."""
    if not breadcrumb or not str(breadcrumb).strip():
        return ""
    parts = [p.strip() for p in str(breadcrumb).split(">") if p.strip()]
    return parts[-1] if parts else ""


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


CHROMA_PAGE_SIZE = 300  # ChromaDB Cloud max results per request


def load_from_chroma(
    collection_name: str,
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    chroma_url: str | None = None,
    chroma_user: str | None = None,
    chroma_password: str | None = None,
) -> tuple[list[dict], list[list[float]] | None, list[str]]:
    """
    Load all products from ChromaDB with metadata, documents, and embeddings.
    Paginates to work around ChromaDB Cloud's 300-result-per-request limit.

    Returns (rows, embeddings, chroma_ids). embeddings may be None if no embeddings.
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
    all_ids: list[str] = []
    all_metadatas: list[dict] = []
    all_documents: list[str] = []
    all_embeddings: list[list[float]] | None = []

    offset = 0
    while True:
        result = collection.get(
            include=["metadatas", "documents", "embeddings"],
            limit=CHROMA_PAGE_SIZE,
            offset=offset,
        )
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or [""] * len(ids)
        embeddings = result.get("embeddings")
        if not ids:
            break
        all_ids.extend(ids)
        all_metadatas.extend(metadatas)
        all_documents.extend(documents)
        if embeddings is not None:
            if all_embeddings is None:
                all_embeddings = []
            all_embeddings.extend(embeddings)
        if len(ids) < CHROMA_PAGE_SIZE:
            break
        offset += len(ids)

    rows = []
    for i, pid in enumerate(all_ids):
        meta = (all_metadatas[i] if i < len(all_metadatas) else {}) or {}
        meta["product_id"] = meta.get("product_id") or pid
        meta["document"] = all_documents[i] if i < len(all_documents) else ""
        meta["name"] = _s(meta.get("name") or meta.get("title"))
        rows.append(meta)

    emb_list = all_embeddings if (all_embeddings and len(all_embeddings) == len(rows)) else None
    return rows, emb_list, list(all_ids)


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


def cluster_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 5,
    cluster_selection_method: str = "leaf",
) -> np.ndarray:
    """HDBSCAN clustering; noise (-1) reassigned to nearest non-noise neighbor.
    Use cluster_selection_method='leaf' for more/finer clusters, 'eom' for fewer.
    """
    try:
        import hdbscan
    except ImportError:
        from sklearn.cluster import KMeans
        k = max(3, min(15, X.shape[0] // 4))
        return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(2, min_cluster_size // 2),
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
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


def _price_tier(p: float) -> str:
    """Price tier label for clustering."""
    if p < 20:
        return "Budget"
    if p < 50:
        return "Mid"
    if p < 120:
        return "Upper Mid"
    return "Premium"


def make_readable_cluster_labels(
    rows: list[dict],
    labels: np.ndarray,
    df: pd.DataFrame | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """
    Generate readable segment_label and subcluster_label for each cluster.
    Returns (segment_labels, subcluster_labels) dicts mapping cluster_id -> str.
    Uses df with text_clean/material when available for richer labels; else uses rows only.
    """
    from collections import Counter
    import re

    segment_labels: dict[int, str] = {}
    subcluster_labels: dict[int, str] = {}

    # Collect cluster members
    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        c = int(lab)
        if c < 0:
            continue
        clusters.setdefault(c, []).append(i)

    for c, indices in sorted(clusters.items()):
        block_rows = [rows[i] for i in indices]
        if not block_rows:
            continue

        # Segment: dominant leaf_category (fallback: derive from category_breadcrumb)
        cats = []
        for r in block_rows:
            cat = (r.get("leaf_category") or r.get("category") or "").strip()
            if not cat:
                cat = _leaf_from_breadcrumb(r.get("category_breadcrumb"))
            if cat:
                cats.append(cat)
        segment = max(set(cats), key=cats.count) if cats else "Mixed"

        # Price tier from median price
        prices = []
        for r in block_rows:
            p = r.get("price_current") or r.get("price") or 0
            try:
                prices.append(float(p))
            except (TypeError, ValueError):
                pass
        med_price = float(np.median(prices)) if prices else 0
        tier = _price_tier(med_price)

        # Subcluster: add material + top product keywords
        if df is not None and "text_clean" in df.columns and len(indices) >= 2:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                block_df = df.iloc[indices]
                vec = TfidfVectorizer(ngram_range=(2, 3), min_df=1)
                X = vec.fit_transform(block_df["text_clean"].fillna(""))
                feats = np.array(vec.get_feature_names_out())
                if len(feats) > 0:
                    mean_vec = np.asarray(X.mean(axis=0)).ravel()
                    top_idx = mean_vec.argsort()[-3:][::-1]
                    top_phrases = [feats[i] for i in top_idx if i < len(feats)][:2]
                    mat = "unknown"
                    if "material" in block_df.columns:
                        modes = block_df["material"].dropna()
                        mat = modes.mode().iloc[0] if len(modes) else "unknown"
                    subcluster_labels[c] = f"{segment} | {mat} | {' / '.join(top_phrases)} | {tier}"
                else:
                    subcluster_labels[c] = f"{segment} | {tier}"
            except Exception:
                subcluster_labels[c] = f"{segment} | {tier}"
        else:
            # Fallback: top words from product names
            words: Counter = Counter()
            for r in block_rows:
                name = _s(r.get("name") or r.get("product_name") or r.get("title"))
                for w in re.findall(r"[a-z0-9]{3,}", name.lower()):
                    if w not in ("new", "day", "thread", "fable", "target", "black", "brown", "pink"):
                        words[w] += 1
            top_words = [w for w, _ in words.most_common(3)][:2]
            kw = " / ".join(top_words) if top_words else ""
            subcluster_labels[c] = f"{segment} | {kw} | {tier}" if kw else f"{segment} | {tier}"

        segment_labels[c] = segment

    return segment_labels, subcluster_labels


def update_chroma_metadata(
    collection_name: str,
    chroma_ids: list[str],
    rows: list[dict],
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    chroma_url: str | None = None,
    chroma_user: str | None = None,
    chroma_password: str | None = None,
) -> None:
    """Persist mission_tags, segment_label, subcluster_label, subcluster to ChromaDB."""
    if not chroma_ids or len(chroma_ids) != len(rows):
        logger.warning("Cannot update ChromaDB: ids/rows length mismatch.")
        return
    client = _get_chroma_client(
        api_key=api_key,
        tenant=tenant,
        database=database,
        chroma_url=chroma_url,
        chroma_user=chroma_user,
        chroma_password=chroma_password,
    )
    collection = client.get_collection(name=collection_name)
    meta_keys = (
        "mission_tags", "mission_primary", "mission_secondary",
        "mission_confidence", "mission_evidence_tokens", "mission_rule_version", "mission_source",
        "segment_label", "subcluster_label", "subcluster",
    )
    metadatas = []
    for r in rows:
        meta = {}
        for k in meta_keys:
            v = r.get(k)
            if v is not None:
                meta[k] = v if isinstance(v, (int, float, bool)) else str(v)
        metadatas.append(meta)
    batch_size = 300
    for i in range(0, len(chroma_ids), batch_size):
        batch_ids = chroma_ids[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        collection.update(ids=batch_ids, metadatas=batch_metas)
    logger.info("Updated ChromaDB: mission_tags, segment_label, subcluster_label, subcluster for %d products.", len(chroma_ids))


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
        "--collection",
        default=os.environ.get("CHROMA_COLLECTION", "target_handbags"),
        help="ChromaDB collection name (default: target_handbags).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input CSV (overrides ChromaDB; use only to load from file instead of DB).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "products_mission_enriched.csv",
        help="Output CSV path (default: output/products_mission_enriched.csv).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid rule-based + LLM fallback tagging (mission_tagging_engine).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM tagging (for testing clustering only; mission_tags will be empty).",
    )
    parser.add_argument(
        "--force-retag",
        action="store_true",
        help="Re-tag all products via LLM even if mission_tags exist in DB.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Alias for --force-retag: re-tag all products via LLM and re-cluster.",
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip clustering (only tag missions).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="HDBSCAN min_cluster_size; lower = more clusters (default: 3).",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["leaf", "eom"],
        default="leaf",
        help="HDBSCAN cluster_selection_method: 'leaf' = more clusters, 'eom' = fewer (default: leaf).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Products per LLM request (default: 20).",
    )
    # ChromaDB args
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
    if args.full_refresh:
        args.force_retag = True

    # Load data: ChromaDB by default (whole dataset), CSV only when -i is explicitly passed
    rows: list[dict] = []
    embeddings: list[list[float]] | np.ndarray | None = None
    from_chroma = args.input is None

    chroma_ids: list[str] = []
    if from_chroma:
        logger.info("Loading from ChromaDB collection '%s' (whole dataset)...", args.collection)
        rows, embeddings, chroma_ids = load_from_chroma(
            collection_name=args.collection,
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

    # Mission tagging: skip products that already have mission_tags in DB (unless --force-retag)
    if not args.skip_llm:
        import sys
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))

        need_tagging = [
            (i, r) for i, r in enumerate(rows)
            if args.force_retag or not _s(r.get("mission_tags"))
        ]
        if need_tagging:
            need_indices = [x[0] for x in need_tagging]
            need_rows = [rows[i] for i in need_indices]
            if args.hybrid:
                from mission_tagging_engine import tag_mission_hybrid_batch
                logger.info("Tagging %d products via hybrid engine (rules + LLM fallback)...", len(need_rows))
                results = tag_mission_hybrid_batch(need_rows)
                for j, i in enumerate(need_indices):
                    res = results[j]
                    rows[i]["mission_tags"] = res.get("mission_tags", "")
                    rows[i]["mission_primary"] = res.get("mission_primary", "")
                    rows[i]["mission_secondary"] = res.get("mission_secondary", "")
                    rows[i]["mission_confidence"] = res.get("mission_confidence", 0)
                    rows[i]["mission_evidence_tokens"] = res.get("mission_evidence_tokens", "")
                    rows[i]["mission_rule_version"] = res.get("mission_rule_version", "")
                    rows[i]["mission_source"] = res.get("mission_source", "")
            else:
                from llm_mission_tagger import tag_missions_batch
                logger.info("Tagging %d products via LLM (batch_size=%d)...", len(need_rows), args.batch_size)
                tag_lists = tag_missions_batch(need_rows, batch_size=args.batch_size)
                for j, i in enumerate(need_indices):
                    rows[i]["mission_tags"] = "|".join(tag_lists[j]) if tag_lists[j] else ""
                # Retry products that still have empty mission_tags (e.g. batch failure or truncation)
                still_empty = [i for i in need_indices if not _s(rows[i].get("mission_tags"))]
                if still_empty:
                    logger.info("Retrying %d products with empty mission_tags (smaller batch)...", len(still_empty))
                    retry_rows = [rows[i] for i in still_empty]
                    retry_tags = tag_missions_batch(retry_rows, batch_size=min(5, len(retry_rows)))
                    for j, i in enumerate(still_empty):
                        rows[i]["mission_tags"] = "|".join(retry_tags[j]) if retry_tags[j] else ""
                    if any(not _s(rows[i].get("mission_tags")) for i in still_empty):
                        logger.warning("Some products still have empty mission_tags after retry.")
        else:
            logger.info("All products already have mission_tags; skipping (use --force-retag to re-tag).")
    else:
        for r in rows:
            r["mission_tags"] = ""

    # Clustering
    label_df: pd.DataFrame | None = None
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
            label_df = df
        labels = cluster_hdbscan(
            X,
            min_cluster_size=args.min_cluster_size,
            cluster_selection_method=args.cluster_method,
        )
        for i, lab in enumerate(labels):
            rows[i]["subcluster"] = int(lab)
        n_clusters = len(set(labels))
        logger.info("Clustering done: %d clusters.", n_clusters)

        # Readable labels for clusters
        segment_labels, subcluster_labels = make_readable_cluster_labels(rows, labels, label_df)
        for i, lab in enumerate(labels):
            c = int(lab)
            rows[i]["segment_label"] = segment_labels.get(c, "")
            rows[i]["subcluster_label"] = subcluster_labels.get(c, "")
    else:
        for r in rows:
            r["subcluster"] = -1
            r["segment_label"] = ""
            r["subcluster_label"] = ""

    # Persist mission_tags and cluster labels to ChromaDB
    if from_chroma and chroma_ids:
        update_chroma_metadata(
            collection_name=args.collection,
            chroma_ids=chroma_ids,
            rows=rows,
            api_key=args.chroma_api_key,
            tenant=args.chroma_tenant,
            database=args.chroma_database,
            chroma_url=args.chroma_url,
            chroma_user=args.chroma_user,
            chroma_password=args.chroma_password,
        )

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

    # Cluster summary (readable segments and subclusters)
    if not args.skip_cluster and rows and "subcluster_label" in rows[0]:
        cluster_summary: list[dict] = []
        for r in rows:
            sc = r.get("subcluster", -1)
            if sc < 0:
                continue
            cluster_summary.append({
                "subcluster": sc,
                "segment_label": r.get("segment_label", ""),
                "subcluster_label": r.get("subcluster_label", ""),
            })
        if cluster_summary:
            seen: set[int] = set()
            unique: list[dict] = []
            for x in cluster_summary:
                sc = x["subcluster"]
                if sc not in seen:
                    seen.add(sc)
                    count = sum(1 for r in rows if r.get("subcluster") == sc)
                    unique.append({
                        "subcluster": sc,
                        "segment_label": x["segment_label"],
                        "subcluster_label": x["subcluster_label"],
                        "product_count": count,
                    })
            summary_path = out_dir / "cluster_summary.csv"
            with open(summary_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["subcluster", "segment_label", "subcluster_label", "product_count"])
                w.writeheader()
                w.writerows(unique)
            logger.info("Wrote %s", summary_path)

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
