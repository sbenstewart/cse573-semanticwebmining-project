import json
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ClassicalRetriever:
    """Implementation of BM25 and TF-IDF baseline ranking."""
    
    def __init__(self, corpus_json_path):
        with open(corpus_json_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        # Extract the text for processing
        # Fallback to 'raw_text' if 'cleaned_text' hasn't been strictly generated yet
        self.corpus_texts = [doc.get('cleaned_text', doc.get('raw_text', '')) for doc in self.documents]
        
        # 1. Initialize BM25
        # BM25 requires tokenized lists of words
        self.tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 2. Initialize TF-IDF
        # TF-IDF uses scikit-learn which handles its own tokenization
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts)

    def search_bm25(self, query, top_k=3):
        """Search using BM25 scoring."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0: # Only return if there's actually a match
                doc = self.documents[idx]
                text_content = doc.get('cleaned_text', doc.get('raw_text', ''))
                results.append({
                    "score": round(scores[idx], 4),
                    "publisher": doc['publisher'],
                    "text_snippet": text_content[:150] + "...",
                    "url": doc['source_url']
                })
        return results

    def search_tfidf(self, query, top_k=3):
        """Search using TF-IDF cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top K indices
        top_n_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if cosine_similarities[idx] > 0:
                doc = self.documents[idx]
                text_content = doc.get('cleaned_text', doc.get('raw_text', ''))
                results.append({
                    "score": round(cosine_similarities[idx], 4),
                    "publisher": doc['publisher'],
                    "text_snippet": text_content[:150] + "...",
                    "url": doc['source_url']
                })
        return results