#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.baseline.topic_model import TopicModel
from src.corpus import load_docs, corpus_stats

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("run_baseline_demo")


def parse_args():
    parser = argparse.ArgumentParser(description="TrendScout AI 2.0 - Baseline Search Demo")
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--mode", choices=["search", "topics", "trends", "stats"], default="search")
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--load-index", action="store_true")
    return parser.parse_args()


def print_results(results: list, method: str) -> None:
    print(f"\n  -- {method.upper()} Results --")
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        date_str = (r.get("published_date") or "")[:10] or "no date"
        print(f"\n  [{i}] {r['title'][:70]}")
        print(f"       Score: {r['score']:.4f}  |  {r['publisher']}  |  {date_str}")
        print(f"       {r['snippet'][:120]}...")
        if r.get("source_url"):
            print(f"       URL: {r['source_url'][:80]}")


def run_search_mode(bm25, tfidf, initial_query, top_k):
    print("\n  TrendScout AI 2.0 - Phase 1 Search Demo")
    print("  Type a query to search, or 'quit' to exit.\n")
    query = initial_query
    while True:
        if query is None:
            try:
                query = input("  Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            query = None
            continue
        print(f"\n  Searching for: '{query}'")
        print_results(bm25.search(query, top_k=top_k), "BM25")
        print_results(tfidf.search(query, top_k=top_k), "TF-IDF")
        print()
        if initial_query is not None:
            break
        query = None


def run_topics_mode(docs):
    print("\n  Fitting LDA topic model...\n")
    tm = TopicModel()
    tm.fit(docs)
    tm.print_topics(top_n=12)
    print("  Top 20 corpus terms:")
    for term, count in tm.top_terms_frequency(top_n=20):
        bar = "█" * min(count // 5, 40)
        print(f"  {term:<20} {count:>5}  {bar}")


def run_stats_mode(docs):
    stats = corpus_stats(docs)
    print(f"\n  CORPUS STATISTICS")
    print(f"  {'─'*45}")
    print(f"  Total documents  : {stats['total_documents']}")
    print(f"  Avg text length  : {stats['avg_text_length']} chars")
    print(f"  Date range       : {stats['date_range']['earliest']} -> {stats['date_range']['latest']}")
    print(f"\n  By publisher:")
    for pub, count in sorted(stats['by_publisher'].items(), key=lambda x: -x[1]):
        print(f"    {pub:<22} {count:>4} docs")


def main():
    args = parse_args()
    print("  Loading corpus...")
    docs = load_docs()
    if not docs:
        print("\n  No documents found.")
        print("  Run: python scripts/run_scraper.py --all")
        print("  Then: python scripts/run_preprocessing.py")
        sys.exit(1)
    print(f"  Loaded {len(docs)} documents\n")

    if args.mode == "stats":
        run_stats_mode(docs)
        return
    if args.mode == "topics":
        run_topics_mode(docs)
        return

    print("  Building search indices...")
    bm25  = BM25Retriever()
    tfidf = TFIDFRetriever()

    if args.load_index:
        try:
            bm25.load()
            tfidf.load()
            print("  Loaded pre-built indices\n")
        except FileNotFoundError:
            bm25.index(docs)
            tfidf.index(docs)
    else:
        bm25.index(docs)
        tfidf.index(docs)
        print("  Indices built\n")

    if args.save_index:
        bm25.save()
        tfidf.save()

    run_search_mode(bm25, tfidf, args.query, args.top_k)


if __name__ == "__main__":
    main()
