import streamlit as st
import os
import json
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document

# --- SET UP PAGE CONFIG ---
st.set_page_config(page_title="RAG vs GraphRAG Evaluator", layout="wide")
st.title("System Comparison: Vector RAG vs GraphRAG")

# --- THE GRAPH PROMPT ---
CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator.

DATABASE SCHEMA:
{schema}

TRANSLATION RULES:
1. You may ONLY use labels and relationships from the schema above.
2. Nodes ONLY have an `id` property. Use case-insensitive matching: `toLower(n.id) CONTAINS 'keyword'`.
3. If the user asks for a simple list of entities (e.g., "what companies"), do NOT invent relationships. Just MATCH the label and RETURN the id. ALWAYS append `LIMIT 50`.
4. NEVER use SQL. Do not use SELECT, JOIN, or subqueries.
5. Output ONLY the raw Cypher query starting with MATCH. No explanations.

The user's question is:
{question}
Cypher: """

CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# --- CACHE THE HEAVY LIFTING ---
@st.cache_resource
def initialize_systems():
    # 1. Init Base LLM
    llm = ChatOllama(model="llama3", temperature=0)
    
    # Variables to track system status
    graph_chain = None
    vector_chain = None
    raw_schema = "Schema not loaded."
    graph_status = "Missing Neo4j Env Vars"
    vector_status = "Not initialized"
    
    # 2. Init GraphRAG (Neo4j)
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    
    if all([neo4j_uri, neo4j_user, neo4j_password]):
        try:
            graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)
            graph.refresh_schema()
            raw_schema = graph.schema
            graph_chain = GraphCypherQAChain.from_llm(
                graph=graph, 
                cypher_llm=llm, 
                qa_llm=llm, 
                cypher_prompt=CYPHER_PROMPT, 
                verbose=True, 
                allow_dangerous_requests=True,
                top_k=10 
            )
            graph_status = "Success ✅"
        except Exception as e:
            graph_status = f"Error: {e} ❌"

    # 3. Init Standard RAG (FAISS Vector Store)
    try:
        # Resolve the path to ../data/master_corpus.json
        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(base_dir, '..', 'data', 'master_corpus.json')
        
        if not os.path.exists(corpus_path):
            vector_status = f"File not found at {corpus_path} ❌"
        else:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
                
            raw_documents = []
            for item in corpus_data:
                # Handle variations in JSON keys ("content", "text", or raw string)
                text = item.get("content", item.get("text", str(item)))
                metadata = {"source": item.get("url", item.get("title", "Unknown Source"))}
                raw_documents.append(Document(page_content=text, metadata=metadata))
                
            # Chunk the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(raw_documents)
            
            # Embed and store
            embeddings = OllamaEmbeddings(model="llama3") 
            vector_store = FAISS.from_documents(docs, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Context: {context}"
            )
            vector_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, vector_prompt)
            vector_chain = create_retrieval_chain(retriever, question_answer_chain)
            vector_status = f"Success ({len(docs)} chunks loaded) ✅"
            
    except Exception as e:
        vector_status = f"Error: {e} ❌"

    return graph_chain, vector_chain, llm, raw_schema, graph_status, vector_status

# --- INITIALIZE APP ---
with st.spinner("Initializing Databases (Embedding JSON might take a minute on first run)..."):
    graph_chain, vector_chain, base_llm, raw_schema, graph_status, vector_status = initialize_systems()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Evaluation Settings")
    query_mode = st.radio(
        "Choose Query Engine:",
        ("Compare Both (Side-by-Side)", "GraphRAG Only (Neo4j)", "Standard RAG Only (FAISS)", "Base LLM Only")
    )
    
    st.markdown("---")
    st.subheader("System Status")
    st.markdown(f"**GraphRAG (Neo4j):** {graph_status}")
    st.markdown(f"**Standard RAG (FAISS):** {vector_status}")
    
    st.markdown("---")
    with st.expander("🔍 View Injected Graph Schema"):
        st.markdown("This is the exact text passed to the LLM for Cypher translation:")
        st.code(raw_schema, language="text")

# --- CHAT HISTORY STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Test a query against the systems..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Querying databases..."):
            
            final_output = ""
            
            # --- EXECUTE VECTOR RAG ---
            if query_mode in ["Standard RAG Only (FAISS)", "Compare Both (Side-by-Side)"]:
                if vector_chain:
                    try:
                        vector_response = vector_chain.invoke({"input": prompt})
                        v_answer = vector_response["answer"]
                        if query_mode == "Compare Both (Side-by-Side)":
                            final_output += f"### 📄 Standard RAG (Vector)\n{v_answer}\n\n---\n\n"
                        else:
                            final_output = v_answer
                    except Exception as e:
                        err_msg = f"**Vector RAG Error:** {e}"
                        if query_mode == "Compare Both (Side-by-Side)":
                            final_output += f"### 📄 Standard RAG (Vector)\n{err_msg}\n\n---\n\n"
                        else:
                            final_output = err_msg
                else:
                    final_output += "### 📄 Standard RAG (Vector)\n*Vector system is offline or failed to load.*\n\n---\n\n"

            # --- EXECUTE GRAPHRAG ---
            if query_mode in ["GraphRAG Only (Neo4j)", "Compare Both (Side-by-Side)"]:
                if graph_chain:
                    try:
                        graph_response = graph_chain.invoke({"query": prompt})
                        g_answer = graph_response['result']
                        if query_mode == "Compare Both (Side-by-Side)":
                            final_output += f"### 🕸️ GraphRAG (Knowledge Graph)\n{g_answer}"
                        else:
                            final_output = g_answer
                    except Exception as e:
                        fallback = "I do not know how to do that yet. My graph query logic failed."
                        if query_mode == "Compare Both (Side-by-Side)":
                            final_output += f"### 🕸️ GraphRAG (Knowledge Graph)\n{fallback}"
                        else:
                            final_output = fallback
                else:
                    final_output += "### 🕸️ GraphRAG (Knowledge Graph)\n*Graph system is offline or missing env vars.*"

            # --- EXECUTE BASE LLM ---
            if query_mode == "Base LLM Only":
                response = base_llm.invoke(prompt)
                final_output = response.content

            st.markdown(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})

if __name__ == "__main__":
    import os
    os.system("streamlit run src/ui.py")