"""Main Streamlit application entrypoint for FEVER IR Fact Verification with Notion Design System."""
import json
from pathlib import Path
import streamlit as st

from src.utils.config_loader import get_storage_dir

st.set_page_config(
    page_title="Information Retrieval System — FEVER",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================================
# NOTION DESIGN SYSTEM GLOBAL CSS
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base Reset & Notion Typography */
html, body, [class*="css"] {
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol" !important;
    color: #E6E6E6 !important;
}

/* Authentic Notion Dark Theme Background */
.stApp {
    background-color: #191919 !important;
    background-image: none !important;
}

/* Ensure Streamlit fixed header integrates seamlessly */
header[data-testid="stHeader"] {
    background-color: #191919 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    z-index: 999990 !important;
}

/* Completely remove sidebar / collapse arrow */
[data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}
div[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Adjust Main block padding */
.block-container {
    padding-top: 4.8rem !important;
    padding-bottom: 4rem !important;
    max-width: 1100px !important;
}

/* =========================================================================
   TOP-LEFT HEADER BAR BRAND (IN SAME LINE AS DEPLOY BUTTON)
   ========================================================================= */
.notion-topbar-left {
    position: fixed;
    top: 0;
    left: 0;
    height: 3.75rem;
    display: flex;
    align-items: center;
    padding-left: 18px;
    z-index: 1000000;
    pointer-events: auto;
    user-select: none;
}
.notion-topbar-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border-radius: 6px;
    background: #202020;
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #FFFFFF;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    transition: all 0.15s ease;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.notion-topbar-pill:hover {
    background: #282828;
    border-color: rgba(255, 255, 255, 0.2);
}
.notion-topbar-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #E6E6E6;
}
.notion-topbar-title {
    color: #FFFFFF;
    font-weight: 600;
    font-size: 0.88rem;
}
.notion-topbar-sep {
    color: rgba(255, 255, 255, 0.25);
    font-size: 0.8rem;
}
.notion-topbar-badge {
    font-size: 0.72rem;
    color: #9B9B9B;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1px 6px;
    border-radius: 4px;
    font-weight: 500;
}

/* =========================================================================
   NOTION PAGE HEADER & PROPERTY LIST
   ========================================================================= */
.notion-page-header {
    margin-bottom: 22px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.notion-page-icon {
    font-size: 42px;
    line-height: 1;
    margin-bottom: 12px;
    display: inline-block;
}
.notion-page-title {
    font-size: 2.15rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.025em;
    line-height: 1.25;
    margin-bottom: 12px;
}
.notion-properties-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 12px;
    font-size: 0.82rem;
    color: #9B9B9B;
    margin-top: 6px;
}
.notion-prop-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border-radius: 5px;
    background: #202020;
    border: 1px solid rgba(255, 255, 255, 0.07);
}
.notion-prop-label {
    color: #787774;
}
.notion-prop-val {
    color: #E6E6E6;
    font-weight: 500;
}

/* =========================================================================
   NOTION VIEW TABS / NAVIGATION
   ========================================================================= */
[data-testid="stPageLink-NavLink"] {
    background: #202020 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
    padding: 8px 14px !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    color: #9B9B9B !important;
    justify-content: center !important;
    transition: all 0.12s ease !important;
    text-align: center !important;
    box-shadow: none !important;
}
[data-testid="stPageLink-NavLink"]:hover {
    background: #262626 !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
    color: #FFFFFF !important;
}
[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: #2C2C2C !important;
    border-color: rgba(255, 255, 255, 0.24) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
}

/* =========================================================================
   NOTION BUTTONS & INPUTS
   ========================================================================= */
.stButton > button {
    background: #252525 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #E6E6E6 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
    transition: background 0.12s ease, border-color 0.12s ease !important;
}
.stButton > button:hover {
    background: #2E2E2E !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"] {
    background: #2383E2 !important;
    border: 1px solid #2383E2 !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2A8DF0 !important;
    border-color: #2A8DF0 !important;
}

/* Text Inputs & Selectboxes */
.stTextInput input, .stTextArea textarea {
    background-color: #202020 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #E6E6E6 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2383E2 !important;
    box-shadow: 0 0 0 1px #2383E2 !important;
}

/* =========================================================================
   NOTION ABOUT SECTION STYLING (REPLACES TELEMETRY CARD)
   ========================================================================= */
.notion-about-container {
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.notion-about-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.notion-about-icon {
    font-size: 1.15rem;
}
/* Notion About Section Styling */
.notion-about-container {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.notion-about-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.notion-about-icon {
    font-size: 1.15rem;
}
.notion-about-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.015em;
}
.notion-about-content {
    font-size: 0.92rem;
    color: #CCCCCC;
    line-height: 1.65;
}
.notion-about-content p {
    margin-bottom: 12px;
}
.notion-about-content ul {
    margin: 8px 0 14px 22px;
    padding: 0;
}
.notion-about-content li {
    margin-bottom: 8px;
    color: #CCCCCC;
}
.notion-about-content strong {
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# 1. FIXED TOP-LEFT HEADER BAR BRAND (SAME LINE AS DEPLOY BUTTON)
# =========================================================================
st.markdown("""
<div class="notion-topbar-left">
    <div class="notion-topbar-pill">
        <span class="notion-topbar-icon">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                <circle cx="12" cy="10" r="3"></circle>
                <line x1="14.5" y1="12.5" x2="17" y2="15"></line>
            </svg>
        </span>
        <span class="notion-topbar-title">Information Retrieval System</span>
        <span class="notion-topbar-sep">/</span>
        <span class="notion-topbar-badge">FEVER</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# 2. NOTION PAGE TITLE & PROPERTY METADATA
# =========================================================================
st.markdown("""
<div class="notion-page-header">
    <div class="notion-page-icon">⚖️</div>
    <div class="notion-page-title">FEVER: An Evidence-Based Information Retrieval System for Automated Fact Verification</div>
    <div class="notion-properties-grid">
        <div class="notion-prop-item">
            <span class="notion-prop-label">🏷️ System:</span>
            <span class="notion-prop-val">Fact Verification &amp; Evidence IR</span>
        </div>
        <div class="notion-prop-item">
            <span class="notion-prop-label">📚 Benchmark:</span>
            <span class="notion-prop-val">FEVER Academic Standard</span>
        </div>
        <div class="notion-prop-item">
            <span class="notion-prop-label">⚙️ IR Engine:</span>
            <span class="notion-prop-val">BM25 Lexical + Live Wikipedia API</span>
        </div>
        <div class="notion-prop-item">
            <span class="notion-prop-label">🟢 Status:</span>
            <span class="notion-prop-val">Calibrated &amp; Grounded</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# 3. NOTION TOP HORIZONTAL TABS (VIEW SWITCHER)
# =========================================================================
pages = [
    st.Page("pages/1_Fact_Verification.py", title="Claim Verifier", icon="⚖️", default=True),
    st.Page("pages/2_IR_Explorer.py", title="Knowledge Base Search", icon="🔎", url_path="IR_Explorer"),
    st.Page("pages/4_Evaluation.py", title="Benchmark Evaluation", icon="📊", url_path="Evaluation"),
    st.Page("pages/3_Dataset_Explorer.py", title="Dataset & Corpus", icon="📚", url_path="Dataset_Explorer"),
]

pg = st.navigation(pages, position="hidden")

t_col1, t_col2, t_col3, t_col4 = st.columns(4)
with t_col1:
    st.page_link(pages[0], label="Claim Verifier", icon="⚖️")
with t_col2:
    st.page_link(pages[1], label="Knowledge Base Search", icon="🔎")
with t_col3:
    st.page_link(pages[2], label="Benchmark Evaluation", icon="📊")
with t_col4:
    st.page_link(pages[3], label="Dataset & Corpus", icon="📚")

st.markdown("<div style='height: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 22px;'></div>", unsafe_allow_html=True)


# =========================================================================
# 4. RUN CURRENT SELECTED FEATURE PAGE
# =========================================================================
pg.run()


# =========================================================================
# 5. ABOUT THE SYSTEM SECTION (SIMPLE TEXT)
# =========================================================================
st.markdown("---")
st.subheader("📖 About this System")
st.markdown(
    """The **Information Retrieval and Fact Verification System** is an academic platform designed to evaluate factual claims using evidence retrieved from Wikipedia.

When a factual claim is entered, the system operates in two core stages:

* **Information Retrieval:** The system searches an indexed Wikipedia knowledge base using the Okapi BM25 ranking algorithm to retrieve relevant evidence sentences matching the subject and key asserted attributes of the claim. Users can also enable live Wikipedia search for real-time document discovery.
* **Fact Verification:** The retrieved evidence sentences are analyzed to determine whether they confirm, contradict, or lack sufficient information about the assertion. If the evidence supports the statement, the verdict is **SUPPORTS**. If the evidence directly contradicts the assertion (such as conflicting numbers, dates, or locations), the verdict is **REFUTES**. If the retrieved passages do not contain enough facts to verify the assertion, the system predicts **NOT ENOUGH INFO**, preventing absence of evidence from being misclassified as contradiction.

Every prediction includes the exact evidence sentences retrieved, the reasoning behind the verdict, and a structured comparison between the claim and the verified Wikipedia record."""
)
