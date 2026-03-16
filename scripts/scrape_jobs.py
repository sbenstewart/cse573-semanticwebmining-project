import requests
import json
import os
import sys
import re
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_models import StartupDocument

def run_job_scraper(companies):
    print("\n--- Scraping Job Boards ---")
    all_documents = []
    
    for company in companies:
        if company['board_type'] == 'greenhouse':
            print(f"Fetching ALL jobs for {company['name']}...")
            url = f"https://boards-api.greenhouse.io/v1/boards/{company['job_board']}/jobs?content=true"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    jobs = data.get('jobs', [])
                    print(f"Found {len(jobs)} jobs. Processing...")
                    
                    for job in jobs: # NO LIMIT
                        raw_html = job.get('content', '')
                        clean_text = re.sub(r'<.*?>', ' ', raw_html)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        
                        doc = StartupDocument(
                            raw_text=f"Job Title: {job.get('title')}. Description: {clean_text}",
                            source_url=job.get('absolute_url'),
                            publisher=company['name']
                        )
                        all_documents.append(doc.to_dict())
            except Exception as e:
                print(f"Error fetching jobs for {company['name']}: {e}")

    with open("data/job_corpus.json", 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=4)