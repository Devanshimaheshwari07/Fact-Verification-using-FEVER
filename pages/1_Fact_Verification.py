"""Interactive Fact Verification Page styled with the Linear.app Design System."""
import re
import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline.fact_verifier import FactVerifierPipeline
from src.utils.database import ExperimentDatabase

# Notion Design System Page Styling
st.markdown("""
<style>
    /* Hero section */
    .hero-container {
        padding: 4px 0 18px 0;
    }
    .hero-pill {
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
    .hero-pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #2383E2;
        box-shadow: 0 0 6px #2383E2;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #9B9B9B;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    /* Notion Cards */
    .linear-card, .notion-card {
        background: #202020;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: border-color 0.12s ease, background 0.12s ease;
    }
    .linear-card:hover, .notion-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        background: #242424;
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #9B9B9B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }

    /* Evidence Items */
    .evidence-card {
        background: #222222;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 0.12s ease;
    }
    .evidence-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        background: #272727;
    }
    .badge-high {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.4);
        color: #4EBA75;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-med {
        background: rgba(223, 171, 1, 0.15);
        border: 1px solid rgba(223, 171, 1, 0.35);
        color: #F5AC33;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-source {
        background: rgba(35, 131, 226, 0.14);
        border: 1px solid rgba(35, 131, 226, 0.3);
        color: #58A6FF;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Verdict Boxes with Clean Notion Style */
    .verdict-refutes {
        background: #2D1A1A;
        border: 1px solid #7D2D2D;
        border-radius: 6px;
        padding: 22px;
        text-align: center;
    }
    .verdict-supports {
        background: #192C1D;
        border: 1px solid #2B6E38;
        border-radius: 6px;
        padding: 22px;
        text-align: center;
    }
    .verdict-nei {
        background: #2D2615;
        border: 1px solid #73591F;
        border-radius: 6px;
        padding: 22px;
        text-align: center;
    }

    /* Component Table in Step 2 */
    .comparison-box {
        background: #202020;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .synthesis-callout {
        background: #242424;
        border-left: 3px solid #2383E2;
        padding: 12px 16px;
        border-radius: 0 4px 4px 0;
        font-size: 0.91rem;
        color: #E6E6E6;
        line-height: 1.5;
        margin-top: 10px;
    }

    /* Telemetry metrics */
    .metric-cell {
        background: #222222;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 10px;
        text-align: center;
    }
    .metric-cell .label {
        font-size: 0.71rem;
        color: #787774;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-cell .val {
        font-size: 1.05rem;
        font-weight: 600;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">
        <span class="hero-pill-dot"></span>
        Fact Verification &amp; Evidence Retrieval Engine
    </div>
    <div class="hero-title">Verify Claims with Wikipedia Grounding</div>
    <div class="hero-subtitle">
        Enter natural-language claims to retrieve candidate evidence, align asserted components with verified facts, and generate a calibrated verdict.
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return FactVerifierPipeline()


pipeline = get_pipeline()
db = ExperimentDatabase()

# Preset benchmark shortcuts styled as Linear command chips
st.markdown("""
<div style="font-size: 0.72rem; font-weight: 600; color: #62666D; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
    Quick Benchmark Examples
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🗼 Eiffel Tower (London)", help="Claim: The Eiffel Tower is located in London. Expected: REFUTES"):
        st.session_state["claim_input"] = "The Eiffel Tower is located in London."
with c2:
    if st.button("💻 Alan Turing (Enigma)", help="Claim: Alan Turing was an English mathematician who broke German codes at Bletchley Park. Expected: SUPPORTS"):
        st.session_state["claim_input"] = "Alan Turing was an English mathematician who broke German codes at Bletchley Park."
with c3:
    if st.button("🚀 Apollo 11 (Aldrin)", help="Claim: Buzz Aldrin was the first person to step on the Moon during Apollo 11. Expected: REFUTES"):
        st.session_state["claim_input"] = "Buzz Aldrin was the first person to step on the Moon during Apollo 11."
with c4:
    if st.button("🏔️ Mount Everest (Earth)", help="Claim: Mount Everest is the highest mountain on Earth above sea level. Expected: SUPPORTS"):
        st.session_state["claim_input"] = "Mount Everest is the highest mountain on Earth above sea level."

default_claim = st.session_state.get(
    "claim_input",
    "The Eiffel Tower is located in London."
)

claim_text = st.text_area(
    "Enter Claim to Verify:",
    value=default_claim,
    height=80,
    help="Type any declarative factual claim."
)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
with col_ctrl1:
    retrieval_method = st.selectbox(
        "Retrieval Algorithm:",
        ["bm25", "tfidf", "hybrid", "boolean", "semantic"],
        index=0,
        help="Information Retrieval ranking model."
    )
with col_ctrl2:
    top_k = st.slider("Evidence Depth (Top-K):", min_value=1, max_value=8, value=4)
with col_ctrl3:
    enable_live_wiki = st.checkbox(
        "🌐 Search Live Wikipedia",
        value=True,
        help="Expands retrieval to live Wikipedia API if local index coverage is insufficient."
    )

verify_clicked = st.button("⚡ Verify Claim", type="primary", use_container_width=True)

if verify_clicked or "last_output" in st.session_state:
    if verify_clicked:
        with st.spinner("Retrieving Wikipedia evidence and performing component analysis..."):
            output = pipeline.verify_claim(
                claim_text=claim_text,
                retrieval_method=retrieval_method,
                top_k=top_k,
                log_to_db=True,
                enable_live_wiki=enable_live_wiki
            )
            st.session_state["last_output"] = output
    else:
        output = st.session_state["last_output"]

    label = output.predicted_label
    confidence = output.confidence
    comp = output.comparison

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ==========================================
    # STEP 1: RETRIEVE EVIDENCE
    # ==========================================
    st.markdown("""
    <div class="step-header">
        <span style="color: #818CF8;">01 /</span> Retrieved Evidence Sentences
    </div>
    """, unsafe_allow_html=True)

    if not output.evidence:
        st.warning("No candidate evidence records found in the knowledge base.")
    else:
        for idx, ev in enumerate(output.evidence, start=1):
            rel_label = "High" if ev.score > 8.0 or idx == 1 else ("Medium" if ev.score > 4.0 else "Low")
            badge_class = "badge-high" if rel_label == "High" else "badge-med"
            source_tag = "Live Wikipedia" if getattr(ev, "retrieval_method", "") == "live_bm25" else "Local Index"

            st.markdown(f"""
            <div class="evidence-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: 700; color: #F7F8F8; font-size: 0.95rem;">#{idx} &bull; {ev.page_title}</span>
                        <span style="color: #8A8F98; font-size: 0.8rem;">(Sentence {ev.sentence_id})</span>
                        <span class="badge-source">{source_tag}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="{badge_class}">Relevance: {rel_label}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #8A8F98;">Score: {ev.score:.2f}</span>
                    </div>
                </div>
                <div style="font-size: 0.98rem; color: #D1D5DB; line-height: 1.55; font-style: italic;">
                    &ldquo;{ev.text}&rdquo;
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # STEP 2: COMPARE CLAIM WITH EVIDENCE
    # ==========================================
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-header">
        <span style="color: #818CF8;">02 /</span> Claim-Evidence Component Alignment
    </div>
    """, unsafe_allow_html=True)

    if comp and comp.rows:
        comp_df = pd.DataFrame(comp.rows)
        st.dataframe(comp_df, width="stretch", hide_index=True)

        if comp.summary:
            st.markdown(f"""
            <div class="synthesis-callout">
                <b>Synthesis:</b> {comp.summary}
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # STEP 3: PRODUCE THE VERDICT
    # ==========================================
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-header">
        <span style="color: #818CF8;">03 /</span> Verification Verdict & Explanation
    </div>
    """, unsafe_allow_html=True)

    v_col1, v_col2 = st.columns([1, 2])

    with v_col1:
        if label == "SUPPORTS":
            st.markdown(f"""
            <div class="verdict-supports">
                <div style="font-size: 0.72rem; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                    Predicted Verdict
                </div>
                <div style="font-size: 2.3rem; font-weight: 800; color: #10B981; letter-spacing: -0.02em; margin-bottom: 4px;">
                    SUPPORTS
                </div>
                <div style="font-size: 1rem; color: #34D399; font-family: 'JetBrains Mono', monospace;">
                    Confidence: {confidence:.1%}
                </div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #10B981; opacity: 0.9;">
                    &#10004; Relevant corroborating evidence found
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif label == "REFUTES":
            st.markdown(f"""
            <div class="verdict-refutes">
                <div style="font-size: 0.72rem; font-weight: 700; color: #FB7185; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                    Predicted Verdict
                </div>
                <div style="font-size: 2.3rem; font-weight: 800; color: #F43F5E; letter-spacing: -0.02em; margin-bottom: 4px;">
                    REFUTES
                </div>
                <div style="font-size: 1rem; color: #FB7185; font-family: 'JetBrains Mono', monospace;">
                    Confidence: {confidence:.1%}
                </div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #F43F5E; opacity: 0.9;">
                    &#10006; Relevant contradictory evidence found
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-nei">
                <div style="font-size: 0.72rem; font-weight: 700; color: #FBBF24; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                    Predicted Verdict
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #F59E0B; letter-spacing: -0.02em; margin-bottom: 4px;">
                    NOT ENOUGH INFO
                </div>
                <div style="font-size: 1rem; color: #FBBF24; font-family: 'JetBrains Mono', monospace;">
                    Confidence: {confidence:.1%}
                </div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #F59E0B; opacity: 0.9;">
                    &#9888; Evidence insufficient in corpus to establish truth
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Linear-styled dark Plotly chart
        probs_df = pd.DataFrame([
            {"Class": k, "Probability": v}
            for k, v in output.probabilities.items()
        ])
        fig = px.bar(
            probs_df,
            x="Probability",
            y="Class",
            orientation="h",
            color="Class",
            color_discrete_map={"SUPPORTS": "#10B981", "REFUTES": "#F43F5E", "NOT ENOUGH INFO": "#F59E0B"},
            range_x=[0, 1.0],
            text="Probability"
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", marker_line_width=0)
        fig.update_layout(
            height=160,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8A8F98", family="Inter, sans-serif"),
            margin=dict(l=0, r=20, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#62666D")),
            yaxis=dict(tickfont=dict(color="#8A8F98"))
        )
        st.plotly_chart(fig, use_container_width=True)

    with v_col2:
        st.markdown("""
        <div style="font-size: 0.72rem; font-weight: 700; color: #818CF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
            Grounded Explanation
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.02rem; color: #E2E8F0; line-height: 1.6; margin-bottom: 16px;'>{output.explanation}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.72rem; font-weight: 700; color: #8A8F98; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
            Extracted Feature Telemetry
        </div>
        """, unsafe_allow_html=True)

        if output.feature_highlights:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-cell">
                    <div class="label">Containment</div>
                    <div class="val">{output.feature_highlights.get('max_claim_containment', 0):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-cell">
                    <div class="label">Jaccard</div>
                    <div class="val">{output.feature_highlights.get('max_jaccard_similarity', 0):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                neg = output.feature_highlights.get("negation_mismatch", 0) > 0.5
                neg_color = "#F43F5E" if neg else "#10B981"
                st.markdown(f"""
                <div class="metric-cell">
                    <div class="label">Neg Conflict</div>
                    <div class="val" style="color: {neg_color};">{'Yes' if neg else 'No'}</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                num = output.feature_highlights.get("numeric_mismatch", 0) > 0.5
                num_color = "#F43F5E" if num else "#10B981"
                st.markdown(f"""
                <div class="metric-cell">
                    <div class="label">Num Conflict</div>
                    <div class="val" style="color: {num_color};">{'Yes' if num else 'No'}</div>
                </div>
                """, unsafe_allow_html=True)
