#!/usr/bin/env python3
"""Build the document vector index for Approach B (GraphRAG).

Reads the processed corpus, embeds each document's text using
sentence-transformers (bge-base-en-v1.5), and stores the embeddings
on the Document nodes in Neo4j via the native vector index.

Usage:
    python scripts/build_vector_index.py [--wipe] [--batch-size 32]

    --wipe    Drop and recreate the vector index before embedding.
              Use this for full rebuilds.

Typical runtime: ~2-3 minutes on a MacBook Air M-series (CPU-only).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CORPUS_FILE
from src.kg.neo4j_client import Neo4jClient
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_vector_index")


def load_corpus(path: str | Path) -> list[dict]:
    """Load JSONL corpus from disk."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def prepare_text(doc: dict) -> str:
    """Build the text to embed for a document.

    Strategy: title + first ~2000 chars of cleaned_text. This captures
    the headline and lead paragraphs, which contain the key facts.
    bge-base-en-v1.5 has a 512-token window (~2000 chars for English),
    so anything beyond that is truncated by the model anyway.
    """
    title = doc.get("title", "").strip()
    body = doc.get("cleaned_text", "") or doc.get("raw_text", "")
    body = body.strip()[:2000]
    if title and body:
        return f"{title}\n\n{body}"
    return title or body or ""


def main():
    parser = argparse.ArgumentParser(
        description="Build document vector index for GraphRAG"
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="Drop and recreate the vector index before embedding",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Embedding batch size (default: 32)",
    )
    args = parser.parse_args()

    # ---- Load corpus ----
    corpus_path = Path(CORPUS_FILE)
    if not corpus_path.exists():
        logger.error(f"Corpus not found at {corpus_path}")
        sys.exit(1)

    docs = load_corpus(corpus_path)
    logger.info(f"Loaded {len(docs)} documents from {corpus_path}")

    # ---- Prepare texts ----
    texts = []
    doc_ids = []
    for doc in docs:
        text = prepare_text(doc)
        if text:
            texts.append(text)
            doc_ids.append(doc["doc_id"])
        else:
            logger.warning(f"Skipping doc {doc.get('doc_id', '?')}: no text")

    logger.info(f"Prepared {len(texts)} documents for embedding")

    # ---- Embed ----
    logger.info("Loading embedding model (first run downloads ~400MB)...")
    embedder = Embedder()
    t0 = time.perf_counter()
    vectors = embedder.embed(texts, batch_size=args.batch_size)
    embed_time = time.perf_counter() - t0
    logger.info(
        f"Embedded {len(vectors)} documents in {embed_time:.1f}s "
        f"(dim={vectors.shape[1]})"
    )

    # ---- Store in Neo4j ----
    with Neo4jClient() as client:
        store = VectorStore(client, dim=embedder.dim)

        if args.wipe:
            logger.info("Wiping existing vector index...")
            store.drop_index()

        logger.info("Creating vector index (if not exists)...")
        store.create_index()

        # Prepare batch items
        items = [
            {"doc_id": did, "embedding": vec.tolist()}
            for did, vec in zip(doc_ids, vectors)
        ]

        logger.info(f"Storing {len(items)} embeddings in Neo4j...")
        t1 = time.perf_counter()
        n_stored = store.store_embeddings_batch(items, batch_size=50)
        store_time = time.perf_counter() - t1
        logger.info(
            f"Stored {n_stored} embeddings in {store_time:.1f}s"
        )

        # ---- Verify ----
        if store.index_exists():
            logger.info(f"Vector index is active")
        else:
            logger.warning("Vector index not found after creation!")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  VECTOR INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Documents embedded : {len(vectors)}")
    print(f"  Embedding dim      : {vectors.shape[1]}")
    print(f"  Embed time         : {embed_time:.1f}s")
    print(f"  Store time         : {store_time:.1f}s")
    print(f"  Total time         : {embed_time + store_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
