import os
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- BULLETPROOF FEW-SHOT CYPHER PROMPT ---
# --- ENTERPRISE SEMANTIC CYPHER PROMPT ---
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator. Your only job is to translate human questions into perfectly formatted Cypher queries.

You must map the user's words to the EXACT Node Labels and Relationship Types provided in the schema below. 

DATABASE SCHEMA:
{schema}

TRANSLATION RULES (Data Dictionary):
1. Node Labels: You may ONLY use labels from the schema (e.g., Organization, Person, Technology). They are Title Case.
2. Identifiers: Nodes ONLY have an `id` property. Use case-insensitive matching: `toLower(n.id) CONTAINS 'keyword'`.
3. Relationships: Map the user's intent to the closest relationship in the schema:
   - If they ask about employment, use `WORKS_FOR` or `EMPLOYS`.
   - If they ask about partnerships/alliances, use `PARTNERED_WITH` or `PARTNERSHIP`.
   - If they ask about rivals/competitors, look for `COMPETED_WITH` or `LOST_BID_TO`.
   - If the intent is entirely unknown or generic (e.g., "associated with", "related to"), DO NOT invent a relationship. Leave it blank like this: `-[r]-`
4. NEVER use SQL. Do not use SELECT, JOIN, or subqueries.
5. If you need to exclude something, use Cypher syntax: `WHERE type(r) <> 'RELATIONSHIP_NAME'`

Output ONLY the raw Cypher query. No explanations, no markdown formatting, no conversational text. Start directly with MATCH.

The user's question is:
{question}
Cypher: """

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

def setup_graph_rag():
    # --- 1. CONNECT TO NEO4J ---
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Error: Missing Neo4j environment variables. Please check your setup.")
        return

    print("Connecting to Neo4j Database...")
    try:
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
        graph.refresh_schema()
        print("Database schema loaded successfully.")
        
        # --- DEBUG: PRINT THE EXACT SCHEMA LANGCHAIN SEES ---
        print("\n" + "*"*50)
        print("[DEBUG] Here is the exact schema LangChain is feeding to Llama 3:")
        print(graph.schema)
        print("*"*50 + "\n")
        
    except Exception as e:
        print(f"Failed to connect to Neo4j. Error: {e}")
        return

    # --- 2. SETUP LOCAL LLM ---
    print("Initializing Local Ollama LLM (Llama 3)...")
    llm = ChatOllama(model="llama3", temperature=0)

    # --- 3. BUILD THE RAG CHAIN ---
    print("Building Graph Cypher QA Chain...")
    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=llm, 
        qa_llm=llm,     
        cypher_prompt=CYPHER_GENERATION_PROMPT, 
        verbose=True, 
        allow_dangerous_requests=True,
        top_k=10 
    )

    print("\n" + "="*50)
    print("Local GraphRAG System Ready! ")
    print("Commands:")
    print("  'toggle' - Switch between GraphRAG and Base LLM")
    print("  'both'   - Run queries side-by-side for direct comparison")
    print("  'exit'   - Quit the program")
    print("="*50 + "\n")

    mode = "both"

    # --- 4. INTERACTIVE CHAT LOOP ---
    while True:
        current_mode_str = "GraphRAG" if mode == "rag" else "Base LLM" if mode == "base" else "Side-by-Side"
        question = input(f"\n[{current_mode_str}] Ask a question: ")
        
        if question.lower() in ['exit', 'quit']:
            print("Shutting down GraphRAG. Goodbye!")
            break
            
        if question.lower() == 'toggle':
            mode = "base" if mode == "rag" else "rag"
            print(f"\n[System] Switched mode to: {'Base LLM (No Context)' if mode == 'base' else 'GraphRAG (Neo4j Context)'}")
            continue

        if question.lower() == 'both':
            mode = "both"
            print("\n[System] Switched mode to: Side-by-Side Comparison")
            continue
            
        try:
            # --- EXECUTE GRAPHRAG ---
            if mode in ["rag", "both"]:
                print("\n" + "-"*50)
                print("Querying Neo4j Graph Database (GraphRAG)...")
                rag_response = chain.invoke({"query": question})
                print(f"\n[GraphRAG Answer]: {rag_response['result']}")
                print("-"*50)
            
            # --- EXECUTE BASE LLM ---
            if mode in ["base", "both"]:
                print("\n" + "-"*50)
                print("Querying Base Llama 3 (No RAG Context)...")
                base_response = llm.invoke(question)
                print(f"\n[Base LLM Answer]: {base_response.content}")
                print("-"*50)
                
        except Exception as e:
            print(f"\n[Error] The model struggled with that query: {e}")

if __name__ == "__main__":
    setup_graph_rag()