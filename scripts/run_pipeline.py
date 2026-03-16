import os
import sys
import json

# Import our scrapers (we will update them to accept targets next)
from scrape_news import run_news_scraper
from scrape_jobs import run_job_scraper
from scrape_yc import run_yc_scraper

# Our 3 deep-dive companies
TARGET_COMPANIES = [
    {"name": "Scale AI", "yc_slug": "scale-ai", "job_board": "scaleai", "board_type": "greenhouse"},
    {"name": "AssemblyAI", "yc_slug": "assemblyai", "job_board": "assemblyai", "board_type": "greenhouse"},
    {"name": "Replit", "yc_slug": "replit", "job_board": "replit", "board_type": "greenhouse"}
]

def merge_corpora():
    """Combines all collected data into the master corpus."""
    print("\n--- Merging Data into Master Corpus ---")
    files = ["data/news_corpus.json", "data/job_corpus.json", "data/yc_corpus.json"]
    all_docs = []
    
    for file in files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                all_docs.extend(docs)
                print(f"Added {len(docs)} records from {file}")
                
    output_path = "data/master_corpus.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=4)
    print(f"Total documents in Master Corpus: {len(all_docs)}")

def main():
    print("=== Starting TrendScout AI Deep Data Collection ===")
    
    # 1. Scrape YC Profiles (Company details, founders, batch)
    run_yc_scraper(TARGET_COMPANIES)
    
    # 2. Scrape Jobs (All open roles for these specific companies)
    run_job_scraper(TARGET_COMPANIES)
    
    # 3. Scrape News & Funding (Using Google News RSS to target them directly)
    run_news_scraper(TARGET_COMPANIES)
    
    # 4. Merge everything
    merge_corpora()
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()