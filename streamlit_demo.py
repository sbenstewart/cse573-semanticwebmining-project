"""
scripts/streamlit_demo.py

Streamlit web UI for the Phase 1 baseline demo.

Features:
  - BM25 vs TF-IDF side-by-side search comparison
  - Corpus statistics dashboard
  - LDA topic visualization
  - Skill trend chart over time

Run with:
    streamlit run scripts/streamlit_demo.py
"""

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

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TrendScout AI 2.0 — Phase 1 Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resource loading ───────────────────────────────────

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

@st.cache_resource(show_spinner="Training LDA topic model (takes ~60 sec)...")
def build_topic_model(docs):
    tm = TopicModel()
    tm.fit(docs)
    return tm


# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Smiley.svg/1200px-Smiley.svg.png",
             width=60)   # Placeholder — replace with project logo
    st.title("TrendScout AI 2.0")
    st.caption("Phase 1 — Classical IR Baselines")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Search Demo", "📊 Corpus Stats", "🧠 Topic Model", "📈 Skill Trends"],
        index=0,
    )

    st.markdown("---")
    st.caption("CSE 573 — Semantic Web Mining\nArizona State University")


# ── Load data ─────────────────────────────────────────────────

docs = load_corpus()

if not docs:
    st.error(
        "No documents found in corpus.\n\n"
        "Run the pipeline first:\n"
        "```\n"
        "python scripts/run_scraper.py --all\n"
        "python scripts/run_preprocessing.py\n"
        "```"
    )
    st.stop()


# ════════════════════════════════════════════════════════════
# Page: Search Demo
# ════════════════════════════════════════════════════════════

if page == "🔍 Search Demo":
    st.title("🔍 Baseline Search — BM25 vs TF-IDF")
    st.markdown(
        "Compare classical retrieval methods side by side. "
        "Both search the same cleaned corpus."
    )

    bm25  = build_bm25(docs)
    tfidf = build_tfidf(docs)

    # Query input
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search query",
            placeholder='e.g. "LLM startup funding 2024"',
            label_visibility="collapsed",
        )
    with col_k:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    # Example queries
    st.caption("Try: ")
    example_cols = st.columns(4)
    examples = [
        "LLM startup funding",
        "vector database investment",
        "AI hiring engineers",
        "open source language model",
    ]
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
                st.subheader(f"{method_name}")
                if not results:
                    st.info("No results found.")
                    return
                for i, r in enumerate(results, 1):
                    with st.expander(
                        f"[{i}] {r['title'][:65]}  —  score: {r['score']:.4f}"
                    ):
                        st.markdown(f"**Publisher:** {r['publisher']}")
                        st.markdown(f"**Date:** {(r.get('published_date') or 'Unknown')[:10]}")
                        st.markdown(f"**Snippet:** {r['snippet'][:300]}...")
                        if r.get("source_url"):
                            st.markdown(f"[View source ↗]({r['source_url']})")

        render_results(bm25_results,  "📘 BM25 (Okapi)",    col_bm25)
        render_results(tfidf_results, "📗 TF-IDF Cosine",   col_tfidf)


# ════════════════════════════════════════════════════════════
# Page: Corpus Stats
# ════════════════════════════════════════════════════════════

elif page == "📊 Corpus Stats":
    st.title("📊 Corpus Statistics")

    stats = corpus_stats(docs)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Documents",  stats["total_documents"])
    k2.metric("Avg Text Length",  f"{stats['avg_text_length']:,} chars")
    k3.metric("With Cleaned Text", stats["documents_with_cleaned_text"])
    k4.metric("Date Range",
              f"{(stats['date_range']['earliest'] or '?')[:7]} → "
              f"{(stats['date_range']['latest'] or '?')[:7]}")

    st.markdown("---")
    st.subheader("Documents by Publisher")

    pub_df = pd.DataFrame(
        list(stats["by_publisher"].items()),
        columns=["Publisher", "Count"]
    ).sort_values("Count", ascending=False)

    st.bar_chart(pub_df.set_index("Publisher"))

    st.markdown("---")
    st.subheader("Sample Documents")
    sample_df = pd.DataFrame([
        {
            "ID": d.get("doc_id", ""),
            "Title": d.get("title", "")[:60],
            "Publisher": d.get("publisher", ""),
            "Date": (d.get("published_date") or "")[:10],
            "Length": len(d.get("cleaned_text") or d.get("raw_text", "")),
        }
        for d in docs[:50]
    ])
    st.dataframe(sample_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# Page: Topic Model
# ════════════════════════════════════════════════════════════

elif page == "🧠 Topic Model":
    st.title("🧠 LDA Topic Discovery")
    st.markdown(
        "Latent Dirichlet Allocation discovers latent topics across the corpus. "
        "Each topic is represented by its highest-probability words."
    )

    if st.button("🚀 Train Topic Model", type="primary"):
        with st.spinner("Training LDA... (may take up to 60 seconds)"):
            tm = build_topic_model(docs)
        st.success("Model trained!")
    else:
        try:
            tm = build_topic_model(docs)
        except Exception:
            st.info("Click 'Train Topic Model' to begin.")
            st.stop()

    topics = tm.get_topics(top_n=12)

    st.subheader(f"Discovered {len(topics)} Topics")

    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            st.markdown(f"**Topic {topic['topic_id']}**")
            word_data = pd.DataFrame(
                list(topic["top_word_probs"].items()),
                columns=["Word", "Probability"]
            )
            st.bar_chart(word_data.set_index("Word"))

    st.markdown("---")
    st.subheader("Query Topic Relevance")
    search_topic_text = st.text_area(
        "Paste or type text to see its topic distribution:",
        height=100,
    )
    if search_topic_text:
        temp_doc = {"doc_id": "query", "cleaned_text": search_topic_text, "raw_text": ""}
        dist = tm.get_document_topics(temp_doc)
        if dist:
            dist_df = pd.DataFrame(dist).sort_values("probability", ascending=False)
            st.bar_chart(dist_df.set_index("topic_id"))
        else:
            st.info("No strong topic signal found in the text.")


# ════════════════════════════════════════════════════════════
# Page: Skill Trends
# ════════════════════════════════════════════════════════════

elif page == "📈 Skill Trends":
    st.title("📈 Skill & Technology Trends")
    st.markdown(
        "Track how frequently AI skills and technologies are mentioned "
        "across the corpus over time."
    )

    # Skill selection
    default_skills = ["LLM", "RAG", "RLHF", "transformer", "fine-tuning"]
    skills = st.multiselect(
        "Select skills to track:",
        options=[
            "LLM", "RAG", "RLHF", "transformer", "fine-tuning",
            "vector database", "diffusion", "multimodal", "PyTorch",
            "CUDA", "reinforcement learning", "computer vision", "NLP",
        ],
        default=default_skills,
    )

    if skills:
        tm = TopicModel()
        tm.docs = docs
        trends = tm.skill_trend_over_time(skills)

        # Build dataframe
        all_months = sorted({
            month
            for skill_data in trends.values()
            for month in skill_data.keys()
            if month != "unknown"
        })

        if all_months:
            trend_df = pd.DataFrame(
                {
                    skill: [trends[skill].get(m, 0) for m in all_months]
                    for skill in skills
                },
                index=all_months,
            )
            st.line_chart(trend_df)

            st.markdown("---")
            st.subheader("Total Mentions by Skill")
            totals = {s: sum(trends[s].values()) for s in skills}
            totals_df = pd.DataFrame(
                list(totals.items()), columns=["Skill", "Total Mentions"]
            ).sort_values("Total Mentions", ascending=False)
            st.bar_chart(totals_df.set_index("Skill"))
        else:
            st.warning(
                "No dated documents found. Dates may be missing from scraped content. "
                "Try running with sources that publish timestamped articles."
            )
    else:
        st.info("Select at least one skill from the dropdown above.")
