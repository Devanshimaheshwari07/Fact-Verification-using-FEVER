"""Dataset Explorer Page for official FEVER dataset and Wikipedia knowledge base with Notion Design System."""
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.config_loader import get_storage_dir
from src.data.load_data import load_fever_claims, stream_wiki_sentences
from src.data.validate_data import validate_claims_dataset

st.markdown("""
<style>
    .data-hero {
        padding: 4px 0 16px 0;
    }
    .data-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #202020;
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
    .data-pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #2383E2;
        box-shadow: 0 0 6px #2383E2;
    }
    .data-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
    }
    .data-subtitle {
        font-size: 0.95rem;
        color: #9B9B9B;
        margin-bottom: 16px;
    }
    .stat-card-linear {
        background: #202020;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 14px;
        text-align: center;
        transition: all 0.12s ease;
    }
    .stat-card-linear:hover {
        background: #242424;
        border-color: rgba(255, 255, 255, 0.16);
    }
    .stat-card-linear .lbl {
        font-size: 0.71rem;
        color: #787774;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .stat-card-linear .num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="data-hero">
    <div class="data-pill">
        <span class="data-pill-dot"></span>
        Official Academic Benchmark Data
    </div>
    <div class="data-title">Dataset &amp; Corpus Explorer</div>
    <div class="data-subtitle">
        Explore 185,445 official FEVER claims, verified evidence annotations, and authentic Wikipedia knowledge base passages.
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_official_claims(split_key: str):
    """Loads claims from the official FEVER dataset files."""
    raw_dir = get_storage_dir("data/raw")
    file_map = {
        "dev": "shared_task_dev.jsonl",
        "train": "train.jsonl",
        "test": "shared_task_test.jsonl"
    }
    target_path = raw_dir / file_map.get(split_key, "shared_task_dev.jsonl")
    if not target_path.exists():
        fallback_path = Path(f"data/sample/{split_key}.jsonl")
        if fallback_path.exists():
            target_path = fallback_path

    return load_fever_claims(str(target_path))


@st.cache_data
def load_official_wiki_articles(max_pages: int = 150):
    """Streams authentic Wikipedia sentences from official FEVER Wikipedia partition files."""
    raw_dir = get_storage_dir("data/raw")
    wiki_dir = raw_dir / "wiki-pages"
    if wiki_dir.exists():
        # Directly target wiki-001.jsonl for fast, permission-safe file access
        w1 = wiki_dir / "wiki-001.jsonl"
        if w1.exists():
            try:
                return list(stream_wiki_sentences(str(w1), max_pages=max_pages))
            except Exception:
                pass
        try:
            return list(stream_wiki_sentences(str(wiki_dir), max_pages=max_pages))
        except Exception:
            pass

    sample_file = Path("data/sample/wiki_sample.jsonl")
    if sample_file.exists():
        return list(stream_wiki_sentences(str(sample_file), max_pages=max_pages))
    return []


tab1, tab2 = st.tabs(["📝 FEVER Claims & Gold Evidence", "📖 Wikipedia Knowledge Base"])

with tab1:
    split_choice = st.radio(
        "Select Official FEVER Split:",
        [
            "⚡ Official Development Split (shared_task_dev.jsonl — 19,998 claims)",
            "🏋️ Official Training Split (train.jsonl — 145,449 claims)",
            "🧪 Official Test Split (shared_task_test.jsonl — 19,998 claims)"
        ],
        index=0,
        horizontal=False
    )

    if "Development" in split_choice:
        split_key = "dev"
        split_name = "Official Development Set"
    elif "Training" in split_choice:
        split_key = "train"
        split_name = "Official Training Set"
    else:
        split_key = "test"
        split_name = "Official Test Set"

    with st.spinner(f"Loading {split_name}..."):
        claims = load_official_claims(split_key)

    stats = validate_claims_dataset(claims)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Total Claims</div>
            <div class="num">{stats['total_claims']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Verifiable Claims</div>
            <div class="num">{stats['claims_with_gold_evidence']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Multi-Hop Evidence</div>
            <div class="num">{stats['claims_with_multi_hop_evidence']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Avg Token Length</div>
            <div class="num">{stats['average_claim_token_length']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Class distribution chart styled with dark theme
    if stats["class_distribution"]:
        st.markdown("""
        <div style="font-size: 0.85rem; font-weight: 600; color: #8A8F98; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
            Official Class Distribution
        </div>
        """, unsafe_allow_html=True)

        dist_df = pd.DataFrame([
            {"Class": k, "Count": v}
            for k, v in stats["class_distribution"].items()
        ])
        fig = px.pie(
            dist_df,
            names="Class",
            values="Count",
            color="Class",
            color_discrete_map={"SUPPORTS": "#10B981", "REFUTES": "#F43F5E", "NOT ENOUGH INFO": "#F59E0B"},
            hole=0.55
        )
        fig.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8A8F98", family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Filter and browse claims
    st.markdown("""
    <div style="font-size: 0.85rem; font-weight: 600; color: #8A8F98; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
        Browse Official Claims
    </div>
    """, unsafe_allow_html=True)

    col_flt1, col_flt2, col_flt3 = st.columns([2, 3, 1])
    with col_flt1:
        labels_present = list(stats["class_distribution"].keys()) if stats["class_distribution"] else ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        selected_label = st.multiselect("Filter by Label:", labels_present, default=labels_present)
    with col_flt2:
        search_query = st.text_input("🔍 Search Claims by Keyword:", placeholder="e.g., Turing, tower, president...")
    with col_flt3:
        max_rows = st.selectbox("Display Rows:", [50, 100, 200, 500], index=1)

    filtered_claims = [c for c in claims if c.label in selected_label]
    if search_query:
        query_lower = search_query.strip().lower()
        filtered_claims = [c for c in filtered_claims if query_lower in c.claim.lower()]

    st.caption(f"Showing {min(len(filtered_claims), max_rows)} of {len(filtered_claims):,} matching claims ({len(claims):,} total in split).")

    claims_table = []
    for c in filtered_claims[:max_rows]:
        gold_ev = c.get_flat_gold_evidence()
        gold_ev_str = "; ".join(f"[{item[0]}: Sent {item[1]}]" for item in list(gold_ev)[:3]) if gold_ev else "None (NEI)"
        if len(gold_ev) > 3:
            gold_ev_str += f" (+{len(gold_ev)-3} more)"

        claims_table.append({
            "ID": c.claim_id,
            "Claim": c.claim,
            "Label": c.label,
            "Verifiable": c.verifiable,
            "Gold Evidence": gold_ev_str
        })

    st.dataframe(pd.DataFrame(claims_table), width="stretch", hide_index=True)

with tab2:
    st.markdown("""
    <div style="font-size: 0.85rem; font-weight: 600; color: #8A8F98; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
        Official FEVER Wikipedia Corpus
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading official Wikipedia sentences..."):
        wiki_sentences = load_official_wiki_articles(max_pages=200)

    unique_docs = sorted(list(set(s.page_title for s in wiki_sentences)))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Sampled Sentences</div>
            <div class="num">{len(wiki_sentences):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Unique Wikipedia Pages</div>
            <div class="num">{len(unique_docs):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        avg_len = sum(len(s.text.split()) for s in wiki_sentences) / len(wiki_sentences) if wiki_sentences else 0
        st.markdown(f"""
        <div class="stat-card-linear">
            <div class="lbl">Avg Sentence Length</div>
            <div class="num">{avg_len:.1f} tokens</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Filter by article
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        selected_doc = st.selectbox("Select Wikipedia Article:", ["All Documents"] + unique_docs)
    with col_w2:
        doc_search = st.text_input("Filter by Keyword:", placeholder="Search text...")

    sentences_display = wiki_sentences
    if selected_doc != "All Documents":
        sentences_display = [s for s in wiki_sentences if s.page_title == selected_doc]
    if doc_search:
        ds_lower = doc_search.lower()
        sentences_display = [s for s in sentences_display if ds_lower in s.text.lower()]

    wiki_table = [
        {
            "Article ID": s.doc_id,
            "Page Title": s.page_title,
            "Sentence #": s.sentence_id,
            "Evidence Text": s.text
        }
        for s in sentences_display[:200]
    ]
    st.dataframe(pd.DataFrame(wiki_table), width="stretch", hide_index=True)
