"""Information Retrieval Explorer Page with Linear.app Design System."""
import time
import re
import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline.fact_verifier import FactVerifierPipeline
from src.retrieval.wiki_live import live_search

st.markdown("""
<style>
    .ir-hero {
        padding: 4px 0 16px 0;
    }
    .ir-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #252525;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.74rem;
        font-weight: 600;
        color: #9B9B9B;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .ir-pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #2383E2;
        box-shadow: 0 0 6px #2383E2;
    }
    .ir-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
    }
    .ir-subtitle {
        font-size: 0.95rem;
        color: #9B9B9B;
        margin-bottom: 16px;
    }
    .ir-result-card {
        background: #202020;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: all 0.12s ease;
    }
    .ir-result-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        background: #242424;
    }
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }
    .comp-col-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ir-hero">
    <div class="ir-pill">
        <span class="ir-pill-dot"></span>
        Multi-Model Information Retrieval
    </div>
    <div class="ir-title">Knowledge Base Search &amp; Ranking</div>
    <div class="ir-subtitle">
        Search evidence sentences across local indexes and live Wikipedia using BM25, TF-IDF, Semantic embeddings, and Boolean logic.
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return FactVerifierPipeline()


pipeline = get_pipeline()

# Query presets
st.markdown("""
<div style="font-size: 0.72rem; font-weight: 600; color: #62666D; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
    Search Shortcuts
</div>
""", unsafe_allow_html=True)

q1, q2, q3, q4 = st.columns(4)
with q1:
    if st.button("🗼 Eiffel Tower Paris"):
        st.session_state["ir_query"] = "Eiffel Tower located in Paris France"
with q2:
    if st.button("💻 Alan Turing Codebreaking"):
        st.session_state["ir_query"] = "Alan Turing Enigma codebreaking Bletchley Park"
with q3:
    if st.button("🚀 Apollo 11 Moon Landing"):
        st.session_state["ir_query"] = "Apollo 11 Neil Armstrong Buzz Aldrin Moon"
with q4:
    if st.button("🏔️ Mount Everest Mountain"):
        st.session_state["ir_query"] = "Mount Everest highest mountain on Earth"

default_query = st.session_state.get(
    "ir_query",
    "Eiffel Tower located in Paris France"
)

query_text = st.text_input(
    "Enter Search Query:",
    value=default_query,
    help="Type any query, keywords, or entity names."
)

tab1, tab2 = st.tabs(["🎯 Search Engine", "⚖️ Multi-Model Benchmarking"])

with tab1:
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        method_options = [
            ("bm25", "BM25 (Probabilistic Sparse Retrieval)"),
            ("tfidf", "TF-IDF (Vector Space Model)"),
            ("semantic", "Sentence Embedding (Dense Vector Retrieval)"),
            ("hybrid", "Hybrid (RRF: Sparse + Dense Fusion)"),
            ("boolean", "Boolean (Exact Match Logic)")
        ]
        selected_method_tuple = st.selectbox(
            "Core Ranking Algorithm:",
            method_options,
            format_func=lambda x: x[1],
            index=0,
            help="Select the retrieval model to score candidate sentences."
        )
        method = selected_method_tuple[0]
    with c2:
        top_k = st.slider("Max Candidates (Top-K):", min_value=1, max_value=15, value=5)
    with c3:
        search_mode = st.radio(
            "Corpus Target:",
            ["🌐 Live Wikipedia", "🏛️ Local Index", "⚡ Auto-Expand"],
            index=0,
            horizontal=True
        )

    search_clicked = st.button("⚡ Execute Search", type="primary", use_container_width=True)

    if search_clicked or "last_ir_results" in st.session_state:
        if search_clicked:
            start_time = time.time()
            results = []

            with st.spinner(f"Ranking documents via {method.upper()} over {search_mode}..."):
                if search_mode == "🌐 Live Wikipedia":
                    results = live_search(query_text, preprocessor=pipeline.preprocessor, top_k=top_k)
                elif search_mode == "🏛️ Local Index":
                    retriever = pipeline.get_retriever(method)
                    results = retriever.retrieve(query_text, top_k=top_k)
                else:  # Auto-Expand
                    retriever = pipeline.get_retriever(method)
                    local_res = retriever.retrieve(query_text, top_k=top_k)
                    if local_res and local_res[0].score >= 3.5:
                        results = local_res
                    else:
                        live_res = live_search(query_text, preprocessor=pipeline.preprocessor, top_k=top_k)
                        results = live_res if live_res else local_res

            latency_ms = (time.time() - start_time) * 1000
            st.session_state["last_ir_results"] = (results, latency_ms, method, search_mode)
        else:
            results, latency_ms, method, search_mode = st.session_state["last_ir_results"]

        st.markdown(f"""
        <div style="display: flex; gap: 12px; align-items: center; margin: 16px 0 12px 0;">
            <span style="font-size: 0.85rem; color: #8A8F98;">Retrieved <b style="color: #F7F8F8;">{len(results)}</b> passages in</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #34D399; background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 4px;">{latency_ms:.1f} ms</span>
            <span style="font-size: 0.75rem; color: #818CF8; background: rgba(94, 106, 210, 0.12); padding: 2px 8px; border-radius: 4px; text-transform: uppercase; font-weight: 600;">{method} &bull; {search_mode}</span>
        </div>
        """, unsafe_allow_html=True)

        if results:
            for r in results:
                source_tag = "Live Wikipedia" if getattr(r, "retrieval_method", "") == "live_bm25" else "Local Index"
                st.markdown(f"""
                <div class="ir-result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div>
                            <span style="font-weight: 700; color: #F7F8F8; font-size: 0.95rem;">#{r.rank} &bull; {r.page_title}</span>
                            <span style="color: #8A8F98; font-size: 0.8rem; margin-left: 8px;">(Sentence {r.sentence_id})</span>
                            <span style="background: rgba(94, 106, 210, 0.12); color: #A5B4FC; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; margin-left: 8px; font-weight: 500;">{source_tag}</span>
                        </div>
                        <div>
                            <span style="font-family: 'JetBrains Mono', monospace; background: rgba(255, 255, 255, 0.05); color: #F7F8F8; font-size: 0.82rem; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.08);">Score: {r.score:.4f}</span>
                        </div>
                    </div>
                    <div style="font-size: 0.95rem; color: #D1D5DB; line-height: 1.55;">
                        {r.text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Exportable dataframe
            data = [
                {
                    "Rank": r.rank,
                    "Score": round(r.score, 4),
                    "Article": r.page_title,
                    "Sentence ID": r.sentence_id,
                    "Text": r.text
                }
                for r in results
            ]
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
        else:
            st.warning("No matching sentences found for this query.")

with tab2:
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 600; color: #F7F8F8; margin-bottom: 4px;">Core Ranking Models Matrix</div>
    <div style="font-size: 0.85rem; color: #8A8F98; margin-bottom: 14px;">Evaluate how TF-IDF, BM25, Sentence Embedding (Dense Vector), and Hybrid (RRF) score the exact same query in parallel.</div>
    """, unsafe_allow_html=True)

    comp_top_k = st.slider("Candidates per method:", min_value=1, max_value=5, value=3, key="comp_top_k")

    if st.button("⚡ Run Multi-Model Comparison", type="secondary"):
        method_keys = ["tfidf", "bm25", "semantic", "hybrid"]
        method_titles = {
            "tfidf": "TF-IDF (Vector Space)",
            "bm25": "BM25 (Sparse Probabilistic)",
            "semantic": "Sentence Embedding (Dense)",
            "hybrid": "Hybrid (RRF Sparse+Dense)"
        }
        cols = st.columns(len(method_keys))

        for idx, m in enumerate(method_keys):
            retriever = pipeline.get_retriever(m)
            start_t = time.time()
            res = retriever.retrieve(query_text, top_k=comp_top_k) if retriever else []
            dur = (time.time() - start_t) * 1000

            with cols[idx]:
                title_label = method_titles.get(m, m.upper())
                st.markdown(f"""
                <div style="font-size: 0.82rem; font-weight: 700; color: #818CF8; text-transform: uppercase; margin-bottom: 4px;">{title_label}</div>
                <div style="font-size: 0.75rem; color: #8A8F98; font-family: 'JetBrains Mono', monospace; margin-bottom: 12px;">{dur:.2f} ms &bull; {len(res)} hits</div>
                """, unsafe_allow_html=True)

                if not res:
                    st.caption("No matches.")
                else:
                    for r in res:
                        st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 6px; padding: 10px; margin-bottom: 8px;">
                            <div style="font-size: 0.82rem; font-weight: 600; color: #F7F8F8; margin-bottom: 2px;">#{r.rank} {r.page_title} (S{r.sentence_id})</div>
                            <div style="font-size: 0.72rem; color: #818CF8; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;">Score: {r.score:.4f}</div>
                            <div style="font-size: 0.8rem; color: #9CA3AF; line-height: 1.4;">&ldquo;{r.text[:85]}...&rdquo;</div>
                        </div>
                        """, unsafe_allow_html=True)
