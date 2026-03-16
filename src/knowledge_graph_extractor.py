import json
import os
from pyvis.network import Network
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain Imports
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# 1. Define Your Exact Schema
ALLOWED_NODES = [
    "ORGANIZATION", "PERSON", "TECHNOLOGY", 
    "INDUSTRY", "PRODUCT_SERVICE", "FINANCIAL_EVENT"
]

def run_llm_extraction():
    # --- 1. SETUP LOCAL OLLAMA LLM ---
    print("Initializing Local Ollama LLM (Llama 3)...")
    llm = ChatOllama(model="llama3", temperature=0) 
    
    # Strict mode and relationships are turned OFF so the local model can breathe
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
    )

    # --- 2. LOAD DATA ---
    print("Loading Master Corpus...")
    corpus_path = "data/master_corpus.json"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Could not find {corpus_path}. Please check your data folder.")
        return
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Convert ALL documents to LangChain format (skipping empty ones)
    langchain_docs = []
    for doc in documents:
        text = doc.get('cleaned_text', doc.get('raw_text', ''))
        if len(text.strip()) > 10:
            langchain_docs.append(Document(page_content=text))

    print(f"Total documents prepared: {len(langchain_docs)}")
    print("Running PARALLEL extraction locally! Hardware is taking the wheel...")
    
    # --- 3. BATCH EXTRACT GRAPH DATA (PARALLEL) ---
    graph_documents = []
    
    # Helper function for the threads to run
    def process_document(doc):
        try:
            return llm_transformer.convert_to_graph_documents([doc])
        except Exception as e:
            print(f"\n[Warning] Error parsing document (Skipping): {e}")
            return []

    # Adjust max_workers based on your machine. 
    # 2 to 4 is usually the sweet spot for a strong local machine running an 8B model.
    MAX_WORKERS = 3 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all 414 documents to the thread pool
        futures = [executor.submit(process_document, doc) for doc in langchain_docs]
        
        # tqdm(as_completed(...)) updates the progress bar as each thread finishes a document
        for future in tqdm(as_completed(futures), total=len(langchain_docs), desc="Extracting Knowledge Graph (Parallel)"):
            result = future.result()
            if result:
                graph_documents.extend(result)

    raw_rel_count = sum([len(g.relationships) for g in graph_documents])
    print(f"\nExtraction Complete! Found total RAW relationships: {raw_rel_count}")
    
    # --- 4. VISUALIZE LOCALLY (HTML) ---
    print("Building local HTML Visualization...")
    G = nx.Graph()
    color_map = {
        "ORGANIZATION": "#e31a1c", "PERSON": "#1f78b4", "TECHNOLOGY": "#33a02c",
        "INDUSTRY": "#6a3d9a", "PRODUCT_SERVICE": "#ff7f00", "FINANCIAL_EVENT": "#b15928"
    }

    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            G.add_node(node.id, title=node.type, color=color_map.get(node.type, "#999999"), size=20)
        for edge in graph_doc.relationships:
            G.add_edge(edge.source.id, edge.target.id, label=edge.type)

    net = Network(height="800px", width="100%", bgcolor="#111111", font_color="white", select_menu=True, filter_menu=True)
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.15, spring_length=200)
    
    output_file = "data/llm_schema_kg.html"
    net.save_graph(output_file)
    print(f"Visualization saved to '{output_file}'.")

    # --- 5. PUSH TO NEO4J DATABASE ---
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    if neo4j_uri and neo4j_user and neo4j_password:
        print("\nConnecting to Neo4j Database...")
        try:
            graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
            print("Pushing extracted nodes and relationships to Neo4j...")
            print("(Note: Neo4j automatically deduplicates identical nodes!)")
            
            graph.add_graph_documents(graph_documents)
            
            db_nodes = graph.query("MATCH (n) RETURN count(n) as node_count")[0]['node_count']
            db_rels = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")[0]['rel_count']
            
            print(f"Successfully populated Neo4j! Current Database Totals: {db_nodes} Nodes, {db_rels} Relationships.")
            
        except Exception as e:
            print(f"Failed to push to Neo4j: {e}")
    else:
        print("\n[Skip] Neo4j credentials not found. Skipping database upload.")

if __name__ == "__main__":
    run_llm_extraction()