#!/usr/bin/env python3
"""
Ingest Target handbags JSON/CSV into Chroma DB with CLIP text embeddings.
Uses Chroma Cloud (CHROMA_API_KEY) and a custom CLIP embedding function.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root for imports when run as script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root so CHROMA_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Import normalize_row from sibling script
try:
    from normalize_dataset import normalize_row
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from normalize_dataset import normalize_row

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default input path (relative to project root)
DEFAULT_INPUT = PROJECT_ROOT / "output" / "target_handbags_20260220_180219.json"
BATCH_SIZE = 32
# CLIP context is 77 tokens; keep document text within ~500 chars to be safe
MAX_DOCUMENT_CHARS = 500

try:
    from analysis.utils import load_products
except ImportError:
    # Fallback when run from elsewhere or analysis not on path
    def load_products(path: Path) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "products" in data:
                return data["products"]
            return [data]
        if suffix == ".jsonl":
            products = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    products.append(json.loads(line))
            return products
        if suffix == ".csv":
            import csv
            # utf-8-sig strips the BOM (\ufeff) that Excel/Windows adds to CSV files
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        raise ValueError(f"Unsupported format: {suffix}. Use .json, .jsonl, or .csv")


def safe_str(v: object) -> str:
    """Coerce value to string for document/metadata."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip()


def build_document_text(product: dict) -> str:
    """Build a single searchable string per product for CLIP (truncated).
    Expects a record already normalized to the canonical schema.
    """
    parts = []
    for key in ("name", "brand", "description", "material_text", "product_details"):
        raw = product.get(key)
        if not raw:
            continue
        s = safe_str(raw)
        if "|" in s:
            s = s.replace("|", " ")
        if s:
            parts.append(s)
    text = " ".join(parts)
    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS].rsplit(" ", 1)[0] or text[:MAX_DOCUMENT_CHARS]
    return text.strip() or ""


def build_metadata(product: dict) -> dict:
    """Build flat metadata for Chroma (scalar values only).
    Expects a record already normalized to the canonical schema.
    """
    meta = {}
    for key in (
        "product_id",
        "name",
        "product_url",
        "brand",
        "price_current",
        "price_original",
        "price_currency",
        "category_breadcrumb",
        "availability_text",
        "average_rating",
        "review_count",
        "is_sale",
        "new_arrival_flag",
        "best_seller_flag",
        "material_text",
        "sku",
    ):
        v = product.get(key)
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            meta[key] = v
        elif isinstance(v, (int, float)):
            meta[key] = v
        else:
            s = safe_str(v)
            if len(s) > 500:
                s = s[:500]
            meta[key] = s
    return meta


def _first_image_url(product: dict) -> str | None:
    """Return the first image URL from canonical image_url or images_list field."""
    # Prefer pre-extracted image_url (canonical schema)
    direct = safe_str(product.get("image_url"))
    if direct and direct.startswith("http"):
        return direct
    # Fall back to parsing images_list (pipe-separated)
    raw = product.get("images_list") or product.get("images")
    if not raw:
        return None
    s = safe_str(raw)
    if not s:
        return None
    first = s.split("|")[0].strip()
    return first if first and first.startswith("http") else None


def prepare_records(products: list[dict]) -> tuple[list[str], list[str], list[dict], list[str | None], list[str]]:
    """Normalize each product to canonical schema, then build Chroma records.
    Returns (ids, documents, metadatas, first_image_urls, skipped_reasons).
    Deduplicates by product_id (keeps first).
    """
    ids = []
    documents = []
    metadatas = []
    first_image_urls: list[str | None] = []
    skipped = []
    seen_ids: set[str] = set()
    for raw in products:
        # ── Normalize to canonical schema first ──────────────────────────
        p = normalize_row(raw)
        pid = safe_str(p.get("product_id"))
        if not pid:
            skipped.append("missing product_id")
            continue
        if pid in seen_ids:
            skipped.append(f"duplicate product_id: {pid}")
            continue
        seen_ids.add(pid)
        doc = build_document_text(p)
        if not doc:
            skipped.append(f"{pid}: empty document text")
            continue
        meta = build_metadata(p)
        ids.append(pid)
        documents.append(doc)
        metadatas.append(meta)
        first_image_urls.append(_first_image_url(p))
    return ids, documents, metadatas, first_image_urls, skipped


# ---------------------------------------------------------------------------
# CLIP embedding function for Chroma
# ---------------------------------------------------------------------------

def _get_clip_embedding_function():
    """Return the CLIP-based embedding function class (lazy import)."""
    import torch
    import clip
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class CLIPTextEmbeddingFunction(EmbeddingFunction):
        """Embed documents using CLIP text encoder (ViT-B/32)."""

        def __init__(self, device: str | None = None):
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = None
            self._preprocess = None

        def _load_model(self):
            if self._model is not None:
                return
            logger.info("Loading CLIP model ViT-B/32 on %s...", self._device)
            self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)
            self._model.eval()

        def __call__(self, input: Documents) -> Embeddings:
            self._load_model()
            if not input:
                return []
            # CLIP tokenize has context_length=77; truncate long text
            texts = []
            for t in input:
                if len(t) > MAX_DOCUMENT_CHARS:
                    t = t[:MAX_DOCUMENT_CHARS].rsplit(" ", 1)[0] or t[:MAX_DOCUMENT_CHARS]
                texts.append(t or " ")
            with torch.no_grad():
                tokens = clip.tokenize(texts, truncate=True).to(self._device)
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings = features.cpu().float().numpy().tolist()
            return embeddings

    return CLIPTextEmbeddingFunction


def compute_text_and_image_embeddings(
    documents: list[str],
    first_image_urls: list[str | None],
    device: str = "cpu",
) -> list[list[float]]:
    """
    Compute CLIP embeddings per record: average of text embedding and image embedding (when URL present).
    Falls back to text-only if image URL missing or download fails.
    """
    import io
    import torch
    import clip
    import requests
    from PIL import Image

    logger.info("Loading CLIP model ViT-B/32 on %s...", device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    embeddings_out: list[list[float]] = []
    for i, (doc, img_url) in enumerate(zip(documents, first_image_urls)):
        # Text embedding
        text = (doc[:MAX_DOCUMENT_CHARS].rsplit(" ", 1)[0] or doc[:MAX_DOCUMENT_CHARS]) if len(doc) > MAX_DOCUMENT_CHARS else doc
        text = text or " "
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(device)
            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # Image embedding when URL available
        if img_url:
            try:
                resp = requests.get(img_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = model.encode_image(img_tensor)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                # Average text + image in same space
                combined = (text_feat + img_feat) / 2.0
                combined = combined / combined.norm(dim=-1, keepdim=True)
                vec = combined.cpu().float().numpy()[0].tolist()
            except Exception as e:
                logger.debug("Image %s failed (%s), using text only: %s", img_url[:50], e, doc[:40])
                vec = text_feat.cpu().float().numpy()[0].tolist()
        else:
            vec = text_feat.cpu().float().numpy()[0].tolist()

        embeddings_out.append(vec)
        if (i + 1) % 10 == 0:
            logger.info("Computed embeddings %d/%d.", i + 1, len(documents))

    return embeddings_out


def run_ingest(
    input_path: Path,
    collection_name: str,
    api_key: str | None,
    tenant: str | None,
    database: str | None,
    batch_size: int,
    use_images: bool = True,
) -> None:
    """Load data, connect to Chroma Cloud, compute text+image embeddings, and add in batches."""
    import chromadb

    products = load_products(input_path)
    logger.info("Loaded %d products from %s", len(products), input_path)
    ids, documents, metadatas, first_image_urls, skipped = prepare_records(products)
    for msg in skipped:
        logger.warning("Skipped: %s", msg)
    if not ids:
        logger.error("No valid records to ingest.")
        return
    logger.info("Prepared %d records for ingestion.", len(ids))

    api_key = api_key or os.environ.get("CHROMA_API_KEY")
    if not api_key:
        raise ValueError(
            "Chroma API key required. Set CHROMA_API_KEY or pass --api-key."
        )
    tenant = tenant or os.environ.get("CHROMA_TENANT")
    database = database or os.environ.get("CHROMA_DATABASE")

    kwargs = {"api_key": api_key}
    if tenant:
        kwargs["tenant"] = tenant
    if database:
        kwargs["database"] = database
    client = chromadb.CloudClient(**kwargs)
    logger.info("Connected to Chroma Cloud.")

    # Compute embeddings (text + first image URL when use_images and URL present)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    image_urls = first_image_urls if use_images else [None] * len(ids)
    embeddings = compute_text_and_image_embeddings(documents, image_urls, device=device)
    logger.info("Computed %d embeddings (text + image where available).", len(embeddings))

    CLIPEmbedding = _get_clip_embedding_function()
    embedding_function = CLIPEmbedding()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"description": "Target handbags with CLIP text+image embeddings"},
    )

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_emb = embeddings[i : i + batch_size]
        collection.add(
            ids=batch_ids,
            embeddings=batch_emb,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        logger.info("Added batch %d–%d (%d items).", i + 1, min(i + batch_size, len(ids)), len(batch_ids))

    logger.info("Ingest complete. Collection '%s' has %d documents.", collection_name, len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Target handbags JSON/CSV into Chroma DB with CLIP embeddings."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to JSON, JSONL, or CSV file (default: output/target_handbags_20260220_180219.json)",
    )
    parser.add_argument(
        "--collection",
        default="target_handbags",
        help="Chroma collection name (default: target_handbags)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CHROMA_API_KEY"),
        help="Chroma Cloud API key (default: CHROMA_API_KEY env)",
    )
    parser.add_argument(
        "--tenant",
        default=os.environ.get("CHROMA_TENANT"),
        help="Chroma tenant (default: CHROMA_TENANT env)",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("CHROMA_DATABASE"),
        help="Chroma database (default: CHROMA_DATABASE env)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for add() (default: %(default)s)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Use text-only embeddings (do not download or use product image URLs)",
    )
    args = parser.parse_args()
    run_ingest(
        input_path=args.input,
        collection_name=args.collection,
        api_key=args.api_key,
        tenant=args.tenant,
        database=args.database,
        batch_size=args.batch_size,
        use_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
