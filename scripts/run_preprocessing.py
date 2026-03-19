#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.deduplicator import Deduplicator
from src.corpus import load_docs, corpus_stats, CORPUS_FILE
import jsonlines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_preprocessing")


def parse_args():
    parser = argparse.ArgumentParser(description="TrendScout AI 2.0 - Preprocessing Pipeline")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input or CORPUS_FILE
    output_path = args.output or input_path

    logger.info(f"Loading corpus from {input_path}")
    docs = load_docs(input_path)

    if not docs:
        logger.error("No documents found. Run run_scraper.py first.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  PREPROCESSING PIPELINE")
    print(f"{'='*55}")
    print(f"  Input documents  : {len(docs)}")

    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_batch(docs)
    print(f"  After cleaning   : {len(cleaned_docs)} ({len(docs) - len(cleaned_docs)} dropped)")

    if not args.skip_dedup:
        deduplicator = Deduplicator(threshold=args.dedup_threshold)
        final_docs = deduplicator.deduplicate(cleaned_docs)
        print(f"  After dedup      : {len(final_docs)} ({len(cleaned_docs) - len(final_docs)} removed)")
    else:
        final_docs = cleaned_docs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for doc in final_docs:
            writer.write(doc)

    stats = corpus_stats(final_docs)
    print(f"\n  FINAL CORPUS STATS")
    print(f"  {'─'*45}")
    print(f"  Total documents  : {stats['total_documents']}")
    print(f"  Avg text length  : {stats['avg_text_length']} chars")
    print(f"  Date range       : {stats['date_range']['earliest']} -> {stats['date_range']['latest']}")
    print(f"\n  By publisher:")
    for pub, count in sorted(stats['by_publisher'].items(), key=lambda x: -x[1]):
        print(f"    {pub:<22} {count:>4} docs")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
