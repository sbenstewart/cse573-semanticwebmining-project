#!/usr/bin/env python3
"""
scripts/run_scraper.py

CLI entry point for Phase 1 data collection.

Usage:
    python scripts/run_scraper.py --all --max-articles 100
    python scripts/run_scraper.py --sources techcrunch venturebeat hackernews
    python scripts/run_scraper.py --sources jobs --max-articles 50
    python scripts/run_scraper.py --dry-run --all
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper.techcrunch import TechCrunchScraper
from src.scraper.venturebeat import VentureBeatScraper
from src.scraper.yc_scraper import YCScraper
from src.scraper.job_scraper import JobScraper
from src.scraper.hackernews_scraper import HackerNewsScraper
from src.scraper.arxiv_scraper import ArXivScraper
from src.corpus import save_docs, save_raw, corpus_stats, load_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_scraper")

SCRAPERS = {
    "techcrunch":  TechCrunchScraper,
    "venturebeat": VentureBeatScraper,
    "yc":          YCScraper,
    "jobs":        JobScraper,
    "hackernews":  HackerNewsScraper,
    "arxiv":       ArXivScraper,
}

# Recommended targets per source for a balanced corpus
RECOMMENDED_TARGETS = {
    "techcrunch":  60,
    "venturebeat": 50,
    "yc":          50,
    "jobs":        80,   # Capped at 10 per company internally
    "hackernews":  120,
    "arxiv":       60,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrendScout AI 2.0 - Data Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources available:
  techcrunch   TechCrunch AI RSS feed
  venturebeat  VentureBeat AI RSS feed
  yc           Y Combinator company directory
  jobs         Greenhouse/Lever AI job boards (capped 10/company)
  hackernews   Hacker News via Algolia API (no auth needed)
  arxiv        arXiv AI/ML papers via official API (no auth needed)

Examples:
  python scripts/run_scraper.py --all
  python scripts/run_scraper.py --sources hackernews arxiv --max-articles 100
  python scripts/run_scraper.py --balanced
        """
    )
    parser.add_argument("--sources", nargs="+", choices=list(SCRAPERS.keys()))
    parser.add_argument("--all", action="store_true", help="Scrape all sources")
    parser.add_argument("--balanced", action="store_true",
                        help="Scrape all sources with recommended targets for balanced corpus")
    parser.add_argument("--max-articles", type=int, default=50,
                        help="Max articles per source (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover URLs only, do not fetch full articles")
    return parser.parse_args()


def run_scraper(name, scraper_cls, max_articles, dry_run):
    logger.info(f"{'='*50}")
    logger.info(f"Starting: {name.upper()} (max={max_articles})")
    scraper = scraper_cls()
    if dry_run:
        urls = scraper.fetch_article_list(max_articles)
        logger.info(f"[Dry Run] {name}: {len(urls)} URLs discovered")
        for url in urls[:3]:
            print(f"  {url}")
        return []
    docs = scraper.scrape(max_articles=max_articles)
    logger.info(f"[{name}] Collected {len(docs)} documents")
    return docs


def main():
    args = parse_args()

    if not args.sources and not args.all and not args.balanced:
        print("Error: specify --sources, --all, or --balanced")
        print("Run with --help for usage.")
        sys.exit(1)

    # Determine sources and targets
    if args.balanced:
        source_targets = RECOMMENDED_TARGETS
    elif args.all:
        source_targets = {s: args.max_articles for s in SCRAPERS}
    else:
        source_targets = {s: args.max_articles for s in args.sources}

    all_docs = []
    for name, target in source_targets.items():
        scraper_cls = SCRAPERS[name]
        try:
            docs = run_scraper(name, scraper_cls, target, args.dry_run)
            if docs:
                save_raw(docs, name)
                all_docs.extend(docs)
        except Exception as e:
            logger.error(f"[{name}] Scraper failed: {e}", exc_info=True)
            logger.warning(f"[{name}] Continuing with other sources...")

    if not args.dry_run and all_docs:
        save_docs(all_docs)
        all_saved = load_docs()
        stats = corpus_stats(all_saved)

        print(f"\n{'='*55}")
        print(f"  CORPUS SUMMARY")
        print(f"{'='*55}")
        print(f"  Total documents  : {stats['total_documents']}")
        print(f"  Avg text length  : {stats['avg_text_length']} chars")
        print(f"  Date range       : {stats['date_range']['earliest']} -> {stats['date_range']['latest']}")
        print(f"\n  By publisher:")
        for pub, count in sorted(stats['by_publisher'].items(), key=lambda x: -x[1]):
            pct = count / stats['total_documents'] * 100
            bar = "█" * min(count // 3, 30)
            print(f"    {pub:<22} {count:>4} docs ({pct:4.1f}%)  {bar}")
        print(f"\n  Saved to: data/processed/corpus.jsonl")
        print(f"{'='*55}\n")

    elif args.dry_run:
        logger.info("Dry run complete.")
    else:
        logger.warning("No documents collected.")


if __name__ == "__main__":
    main()
