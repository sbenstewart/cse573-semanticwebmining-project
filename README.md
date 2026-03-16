# Local LLM Knowledge Graph & GraphRAG Evaluation Pipeline

This project is an automated, fully local pipeline that takes a corpus of unstructured text documents, uses a Large Language Model (LLM) to extract structured entities and relationships, pushes the graph to a Neo4j database, and provides an interactive UI to evaluate **Standard Vector RAG vs. GraphRAG** side-by-side.

Currently, the project is complete through **Phase 4: Evaluation & System Comparison**.

## Architecture & Tech Stack
* **Local LLM Engine:** Ollama (Running `llama3` for generation/extraction and `nomic-embed-text` for lightning-fast vector embeddings)
* **Orchestration:** LangChain (`langchain-ollama`, `langchain-neo4j`, `langchain-community`)
* **Graph Database:** Neo4j (AuraDB or Local Desktop)
* **Vector Database:** FAISS (Local in-memory vector store for standard RAG)
* **User Interface:** Streamlit
* **Visualization:** NetworkX & Pyvis (for local `.html` graph rendering)
* **Concurrency:** Python `concurrent.futures` (ThreadPoolExecutor for parallel document processing)

---

## Prerequisites & Setup

### 1. Install Ollama and the Models
Because this pipeline runs 100% locally to avoid API costs and data privacy issues, you must have the Ollama engine installed.

**Installation by Operating System:**
* **macOS:** Download the app from [ollama.com](https://ollama.com/) or install via Homebrew:
  ```bash
  brew install ollama
  ```
* **Linux:** Run the automated install script:
  ```bash
  curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
  ```
* **Windows:** Download the official Windows installer directly from [ollama.com](https://ollama.com/).

**Pull the Required Models:**
Once Ollama is installed, open your terminal and download the generative model and the embedding model:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Python Environment Setup
Ensure you have Python 3.9+ installed, then install the required dependencies:
```bash
pip install langchain langchain-ollama langchain-neo4j langchain-community langchain-text-splitters pyvis networkx tqdm streamlit faiss-cpu
```

### 3. Neo4j Environment Variables
The script requires a connection to a Neo4j database to store and query the Knowledge Graph. Set the following environment variables in your terminal or `.env` file prior to running the scripts:
```bash
export NEO4J_URI="bolt://<your-neo4j-uri>"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="<your-password>"
```

---

## Pipeline Phases

### Phase 1: Data Preparation
* **Action:** Cleans and formats raw text documents into a unified structure.
* **Output:** A structured JSON file located at `data/master_corpus.json`. 
* **Format:** The JSON contains an array of document objects containing the text to be processed.

### Phase 2: Knowledge Graph Extraction
The core extraction script (`src/knowledge_graph_extractor.py`) performs the following operations:
1. **Data Loading:** Reads `master_corpus.json`.
2. **Parallel Processing:** Uses a `ThreadPoolExecutor` to process multiple documents concurrently, maximizing local hardware utilization.
3. **LLM Extraction:** Passes text to `llama3` to extract Nodes (e.g., `ORGANIZATION`, `PERSON`, `TECHNOLOGY`) and dynamically determine relationships.
4. **Local HTML Visualization:** Generates a lightweight, interactive web graph saved to `data/llm_schema_kg.html`.
5. **Neo4j Database Push:** Uploads the extracted graph to Neo4j, using `MERGE` operations to deduplicate nodes and relationships into a unified Knowledge Graph.

### Phase 3 & 4: Interactive UI & GraphRAG Evaluation
The user interface (`src/ui.py`) provides a testing ground to compare traditional vector-based retrieval against graph-based retrieval:
1. **Standard RAG (FAISS):** Loads `master_corpus.json`, chunks the text using `RecursiveCharacterTextSplitter`, embeds it using `nomic-embed-text`, and retrieves relevant text blocks via similarity search.
2. **GraphRAG (Neo4j):** Intercepts user questions, uses `llama3` to dynamically translate the natural language into Neo4j Cypher queries (enforcing strict schema rules and fail-safes), and retrieves exact relational data.
3. **Side-by-Side Comparison:** Allows the user to toggle between "Standard RAG Only", "GraphRAG Only", or "Compare Both" to evaluate how each system handles multi-hop reasoning and factual recall.

---

## How to Run

### Step 1: Run Data Preparation (Phase 1)
Process your raw files to generate the required master JSON file:
```bash
python src/data_prep.py
```

### Step 2: Run Knowledge Graph Extraction (Phase 2)
Ensure your Ollama server is running. Execute the extraction script to build your graph in Neo4j:
```bash
python src/knowledge_graph_extractor.py
```

### Step 3: Launch the Evaluation UI (Phase 3 & 4)
Boot up the Streamlit interface to query your data and compare the systems:
```bash
streamlit run src/ui.py
```
*(Note: On the first run, it may take a few moments to chunk and embed the JSON corpus into the FAISS vector database).*

---

## Viewing the Graph directly
If you want to view the raw graph outside of the QA application:
* **Local File:** Open `data/llm_schema_kg.html` in your web browser.
* **Database Query:** Open your Neo4j console and run:
  ```cypher
  MATCH (n)-[r]->(m)
  RETURN n, r, m
  ```