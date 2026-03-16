import sys
import os
import json

# Add the parent directory to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.baselines import ClassicalRetriever

def run_baseline_demo():
    corpus_path = "data/master_corpus.json"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Could not find {corpus_path}. Please run the pipeline first.")
        return

    print(f"Loading Master Corpus and Initializing BM25 & TF-IDF Retrievers...")
    retriever = ClassicalRetriever(corpus_path)
    
    # We test standard queries and your "trick" query
    test_queries = [
        "What is Scale AI funding and who are their investors?",
        "AssemblyAI speech recognition models and features",
        "Replit job openings for software engineers" # The trick question
    ]
    
    for query in test_queries:
        print(f"\n" + "="*50)
        print(f"QUERY: '{query}'")
        print("="*50)
        
        print("\n--- Top 3 BM25 Results ---")
        bm25_results = retriever.search_bm25(query, top_k=3)
        if not bm25_results:
            print("No matching documents found.")
        for i, res in enumerate(bm25_results, 1):
            print(f"{i}. [Score: {res['score']}] Publisher: {res['publisher']}")
            print(f"   Snippet: {res['text_snippet']}")
            
        print("\n--- Top 3 TF-IDF Results ---")
        tfidf_results = retriever.search_tfidf(query, top_k=3)
        if not tfidf_results:
            print("No matching documents found.")
        for i, res in enumerate(tfidf_results, 1):
            print(f"{i}. [Score: {res['score']}] Publisher: {res['publisher']}")
            print(f"   Snippet: {res['text_snippet']}")

if __name__ == "__main__":
    run_baseline_demo()