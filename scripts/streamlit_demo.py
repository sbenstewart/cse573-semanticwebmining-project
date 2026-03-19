import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.baseline.topic_model import TopicModel
from src.corpus import load_docs, corpus_stats

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="TrendScout AI 2.0 - Phase 1 Demo",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Loading corpus...")
def load_corpus():
    return load_docs()

@st.cache_resource(show_spinner="Building BM25 index...")
def build_bm25(docs):
    r = BM25Retriever()
    r.index(docs)
    return r

@st.cache_resource(show_spinner="Building TF-IDF index...")
def build_tfidf(docs):
    r = TFIDFRetriever()
    r.index(docs)
    return r

@st.cache_resource(show_spinner="Training LDA topic model...")
def build_topic_model(docs):
    tm = TopicModel()
    tm.fit(docs)
    return tm

with st.sidebar:
    st.title("TrendScout AI 2.0")
    st.caption("Phase 1 - Classical IR Baselines")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Search Demo", "Corpus Stats", "Topic Model", "Skill Trends"],
        index=0,
    )
    st.markdown("---")
    st.caption("CSE 573 - Semantic Web Mining\nArizona State University")

docs = load_corpus()

if not docs:
    st.error(
        "No documents found.\n\n"
        "Run the pipeline first:\n"
        "```\npython scripts/run_scraper.py --all\n"
        "python scripts/run_preprocessing.py\n```"
    )
    st.stop()

if page == "Search Demo":
    st.title("Baseline Search - BM25 vs TF-IDF")
    bm25  = build_bm25(docs)
    tfidf = build_tfidf(docs)
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input("Search query", placeholder='e.g. LLM startup funding 2024', label_visibility="collapsed")
    with col_k:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
    st.caption("Try:")
    example_cols = st.columns(4)
    examples = ["LLM startup funding", "vector database investment", "AI hiring engineers", "open source language model"]
    for i, ex in enumerate(examples):
        if example_cols[i].button(ex, key=f"ex_{i}"):
            query = ex
    if query:
        st.markdown("---")
        col_bm25, col_tfidf = st.columns(2)
        bm25_results  = bm25.search(query, top_k=top_k)
        tfidf_results = tfidf.search(query, top_k=top_k)
        def render_results(results, method_name, container):
            with container:
                st.subheader(method_name)
                if not results:
                    st.info("No results found.")
                    return
                for i, r in enumerate(results, 1):
                    with st.expander(f"[{i}] {r['title'][:65]}  -  score: {r['score']:.4f}"):
                        st.markdown(f"**Publisher:** {r['publisher']}")
                        st.markdown(f"**Date:** {(r.get('published_date') or 'Unknown')[:10]}")
                        st.markdown(f"**Snippet:** {r['snippet'][:300]}...")
                        if r.get("source_url"):
                            st.markdown(f"[View source]({r['source_url']})")
        render_results(bm25_results,  "BM25 (Okapi)",  col_bm25)
        render_results(tfidf_results, "TF-IDF Cosine", col_tfidf)

elif page == "Corpus Stats":
    st.title("Corpus Statistics")
    stats = corpus_stats(docs)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Documents",   stats["total_documents"])
    k2.metric("Avg Text Length",   f"{stats['avg_text_length']:,} chars")
    k3.metric("With Cleaned Text", stats["documents_with_cleaned_text"])
    k4.metric("Date Range",
              f"{(stats['date_range']['earliest'] or '?')[:7]} -> "
              f"{(stats['date_range']['latest'] or '?')[:7]}")
    st.markdown("---")
    st.subheader("Documents by Publisher")
    pub_df = pd.DataFrame(list(stats["by_publisher"].items()), columns=["Publisher", "Count"]).sort_values("Count", ascending=False)
    st.bar_chart(pub_df.set_index("Publisher"))
    st.subheader("Sample Documents")
    sample_df = pd.DataFrame([{
        "ID": d.get("doc_id", ""),
        "Title": d.get("title", "")[:60],
        "Publisher": d.get("publisher", ""),
        "Date": (d.get("published_date") or "")[:10],
        "Length": len(d.get("cleaned_text") or d.get("raw_text", "")),
    } for d in docs[:50]])
    st.dataframe(sample_df, use_container_width=True)

elif page == "Topic Model":
    st.title("LDA Topic Discovery")
    if st.button("Train Topic Model", type="primary"):
        tm = build_topic_model(docs)
        st.success("Model trained!")
    else:
        try:
            tm = build_topic_model(docs)
        except Exception:
            st.info("Click Train Topic Model to begin.")
            st.stop()
    topics = tm.get_topics(top_n=12)
    st.subheader(f"Discovered {len(topics)} Topics")
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            st.markdown(f"**Topic {topic['topic_id']}**")
            word_data = pd.DataFrame(list(topic["top_word_probs"].items()), columns=["Word", "Probability"])
            st.bar_chart(word_data.set_index("Word"))

elif page == "Skill Trends":
    st.title("Skill and Technology Trends")
    default_skills = ["LLM", "RAG", "RLHF", "transformer", "fine-tuning"]
    skills = st.multiselect(
        "Select skills to track:",
        options=["LLM", "RAG", "RLHF", "transformer", "fine-tuning",
                 "vector database", "diffusion", "multimodal", "PyTorch",
                 "CUDA", "reinforcement learning", "computer vision", "NLP"],
        default=default_skills,
    )
    if skills:
        tm = TopicModel()
        tm.docs = docs
        trends = tm.skill_trend_over_time(skills)
        all_months = sorted({m for sd in trends.values() for m in sd.keys() if m != "unknown"})
        if all_months:
            trend_df = pd.DataFrame(
                {skill: [trends[skill].get(m, 0) for m in all_months] for skill in skills},
                index=all_months,
            )
            st.line_chart(trend_df)
            st.subheader("Total Mentions by Skill")
            totals_df = pd.DataFrame(
                [{"Skill": s, "Total Mentions": sum(trends[s].values())} for s in skills]
            ).sort_values("Total Mentions", ascending=False)
            st.bar_chart(totals_df.set_index("Skill"))
        else:
            st.warning("No dated documents found in corpus.")
