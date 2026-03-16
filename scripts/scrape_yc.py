import requests
from bs4 import BeautifulSoup
import time
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_models import StartupDocument

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def run_yc_scraper(companies):
    print("\n--- Scraping YC Profiles ---")
    all_documents = []
    
    for company in companies:
        slug = company['yc_slug']
        print(f"Fetching YC data for: {company['name']}")
        url = f"https://www.ycombinator.com/companies/{slug}"
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                description = " ".join([p.get_text(strip=True) for p in paragraphs])
                
                raw_text = f"Startup Name: {company['name']}. YC Profile: {description}"
                doc = StartupDocument(raw_text=raw_text, source_url=url, publisher="Y Combinator")
                all_documents.append(doc.to_dict())
            time.sleep(2)
        except Exception as e:
            print(f"Failed to scrape YC for {company['name']}: {e}")

    os.makedirs("data", exist_ok=True)
    with open("data/yc_corpus.json", 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=4)