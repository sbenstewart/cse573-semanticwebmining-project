#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper.techcrunch import TechCrunchScraper
from src.scraper.venturebeat import VentureBeatScraper
from src.scraper.yc_scraper import YCScraper
from src.scraper.job_scraper import JobScraper
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
}


def parse_args():
    parser = argparse.ArgumentParser(description="TrendScout AI 2.0 - Phase 1 Data Scraper")
    parser.add_argument("--sources", nargs="+", choices=list(SCRAPERS.keys()))
    parser.add_argument("--all", action="store_true", help="Scrape all sources")
    parser.add_argument("--max-articles", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_scraper(name, scraper_cls, max_articles, dry_run):
    logger.info(f"{'='*50}")
    logger.info(f"Starting scraper: {name.upper()} (max={max_articles})")
    scraper = scraper_cls()
    if dry_run:
        urls = scraper.fetch_article_list(max_articles)
        logger.info(f"[Dry Run] {name}: {len(urls)} URLs discovered")
        for url in urls[:5]:
            print(f"  {url}")
        return []
    docs = scraper.scrape(max_articles=max_articles)
    logger.info(f"[{name}] Collected {len(docs)} documents")
    return docs


def main():
    args = parse_args()
    if not args.sources and not args.all:
        print("Error: specify --sources or --all")
        sys.exit(1)

    sources = list(SCRAPERS.keys()) if args.all else args.sources
    all_docs = []

    for name in sources:
        try:
            docs = run_scraper(name, SCRAPERS[name], args.max_articles, args.dry_run)
            if docs:
                save_raw(docs, name)
                all_docs.extend(docs)
        except Exception as e:
            logger.error(f"[{name}] Scraper failed: {e}", exc_info=True)

    if not args.dry_run and all_docs:
        save_docs(all_docs)
        stats = corpus_stats(load_docs())
        print(f"\n{'='*50}")
        print(f"  Total documents  : {stats['total_documents']}")
        print(f"  Avg text length  : {stats['avg_text_length']} chars")
        print(f"  Date range       : {stats['date_range']['earliest']} -> {stats['date_range']['latest']}")
        print(f"\n  By publisher:")
        for pub, count in sorted(stats['by_publisher'].items(), key=lambda x: -x[1]):
            print(f"    {pub:<20} {count:>4} docs")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
