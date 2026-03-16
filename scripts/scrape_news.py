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

def run_news_scraper(companies):
    print("\n--- Scraping Company-Specific News & Funding ---")
    all_documents = []
    
    for company in companies:
        name = company['name']
        print(f"Searching news for: {name}...")
        
        # Use Google News RSS targeted exactly at the company name
        feed_url = f"https://news.google.com/rss/search?q=\"{name}\"+startup+OR+funding+OR+AI&hl=en-US&gl=US&ceid=US:en"
        
        try:
            response = requests.get(feed_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            print(f"Found {len(items)} news items for {name}. Processing...")
            
            for item in items: # NO LIMIT
                title = item.title.text if item.title else "No Title"
                link = item.link.text if item.link else None
                pub_date = item.pubDate.text if item.pubDate else datetime.now().isoformat()
                
                # For Google News, we store the snippet as the raw text since following 
                # random news links can hit paywalls or bot-protections.
                description = item.description.text if item.description else title
                clean_desc = BeautifulSoup(description, 'html.parser').get_text(strip=True)
                
                if link and len(clean_desc) > 20:
                    doc = StartupDocument(
                        raw_text=f"Headline: {title}. Summary: {clean_desc}",
                        source_url=link,
                        publisher="Google News / Various",
                        published_date=pub_date
                    )
                    all_documents.append(doc.to_dict())
            time.sleep(2)
        except Exception as e:
            print(f"Error fetching news for {name}: {e}")

    with open("data/news_corpus.json", 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=4)