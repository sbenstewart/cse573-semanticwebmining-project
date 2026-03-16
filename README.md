# Local LLM Knowledge Graph Extraction Pipeline

This project is an automated pipeline that takes a corpus of unstructured text documents and uses a local Large Language Model (LLM) to extract structured entities (Nodes) and relationships (Edges), automatically building a localized HTML visualizer and pushing the deduplicated graph to a Neo4j database.

Currently, the project is complete through **Phase 2: Knowledge Graph Extraction**.

## Architecture & Tech Stack
* **Local LLM Engine:** Ollama (Running the 8B parameter `llama3` model)
* **Orchestration:** LangChain (`langchain-ollama`, `langchain-neo4j`)
* **Graph Database:** Neo4j (AuraDB or Local Desktop)
* **Visualization:** NetworkX & Pyvis (for local `.html` graph rendering)
* **Concurrency:** Python `concurrent.futures` (ThreadPoolExecutor for parallel document processing)

---

## Prerequisites & Setup

### 1. Install Ollama and the Model
Because this pipeline runs 100% locally to avoid API costs and rate limits, you must have the Ollama engine installed on your machine.

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

**Pull the Llama 3 Model:**
Once Ollama is installed, open your terminal and download the specific Llama 3 model we are using:
```bash
ollama pull llama3
```

### 2. Python Environment Setup
Ensure you have Python 3.9+ installed, then install the required dependencies:
```bash
pip install langchain-ollama langchain-neo4j pyvis networkx tqdm
```

### 3. Neo4j Environment Variables
The script pushes the extracted Knowledge Graph to a Neo4j database. You must set the following environment variables in your terminal or `.env` file prior to running the script:
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
* **Format:** The JSON contains an array of document objects, which must include a `cleaned_text` or `raw_text` key.

### Phase 2: Knowledge Graph Extraction
The core script (`src/knowledge_graph_extractor.py`) performs the following operations:
1. **Data Loading:** Reads `master_corpus.json` and converts it into LangChain `Document` objects.
2. **Parallel Processing:** Uses a `ThreadPoolExecutor` to process multiple documents concurrently, pushing the local hardware to 100% utilization to maximize extraction speed.
3. **LLM Extraction:** Passes the text to local `llama3`. 
   * *Note on Schema:* We define strict allowed Nodes (e.g., `ORGANIZATION`, `PERSON`, `TECHNOLOGY`), but we intentionally leave the relationship schema open (disabling LangChain's `strict_mode`). This allows the smaller 8B model to "breathe" and naturally extract connections rather than dropping them when forced into rigid categories.
4. **Local HTML Visualization:** Generates a lightweight, interactive web graph saved to `data/llm_schema_kg.html`.
5. **Neo4j Database Push:** Uploads the extracted `graph_documents` to Neo4j. Neo4j automatically uses a `MERGE` operation to deduplicate overlapping nodes and relationships, resulting in a clean, unified, and highly-connected Knowledge Graph.

---

## How to Run

### Step 1: Run Data Preparation (Phase 1)
First, process your raw files to generate the required master JSON file:
```bash
python src/data_prep.py
```
*(Note: If your data preparation script has a different name, replace `src/data_prep.py` with the correct path).*

### Step 2: Run Knowledge Graph Extraction (Phase 2)
Ensure your Ollama server is running in the background (e.g., `brew services start ollama` or by opening the Ollama app). Then, execute the extraction script:
```bash
python src/knowledge_graph_extractor.py
```
Monitor the terminal for the progress bar. Upon completion, the script will output the total number of deduplicated Nodes and Relationships successfully stored in the Neo4j database.

---

## Viewing the Graph
* **Quick Visual Check:** Open `data/llm_schema_kg.html` in your web browser.
* **Database Query:** Open your Neo4j console and run the following Cypher query to view your connected graph:
   ```cypher
   MATCH (n)-[r]->(m)
   RETURN n, r, m
   ```