# FEVER: An Evidence-Based Information Retrieval System for Automated Fact Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](tests/)
[![FEVER Score](https://img.shields.io/badge/FEVER%20Score-83.33%25-success.svg)](artifacts/results/evaluation_results.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academically rigorous, end-to-end Information Retrieval and Fact Verification system implementing the official **Fact Extraction and VERification (FEVER)** pipeline. Given a natural language factual claim, the system retrieves relevant evidence from an indexed Wikipedia knowledge base and determines whether the claim is **SUPPORTS**, **REFUTES**, or **NOT ENOUGH INFO** with transparent, evidence-grounded justifications.

---

## 1. Project Abstract

In an era characterized by the rapid dissemination of unverified information, automated fact verification is a foundational capability for information retrieval and computational journalism. This project constructs an evidence-grounded Information Retrieval system for automated fact verification based on the official FEVER benchmark (Thorne et al., 2018). Rather than treating verification as an ungrounded black-box classification task, this system places primary academic emphasis on the Information Retrieval pipeline: document preprocessing, inverted index construction, probabilistic BM25 ranking, vector space TF-IDF matching, and hybrid rank fusion. Retrieved sentence candidates are passed to an interpretable verification engine that extracts linguistic, containment, negation, and numeric signals to classify claims and output evidence-grounded reasoning. On the held-out development evaluation, the system achieves **0.8889 Mean Reciprocal Rank (MRR)**, **94.44% Recall@5**, and an official **FEVER Score of 83.33%** with sub-millisecond retrieval latency.

---

## 2. Problem Statement

Given:
1. A natural language claim $C$ (e.g., *"Alan Turing was an English mathematician who broke German codes at Bletchley Park"*).
2. A reference knowledge base $\mathcal{K}$ consisting of Wikipedia sentences.

The objective is twofold:
- **Evidence Retrieval**: Select a compact subset of evidence sentences $\mathcal{E} \subseteq \mathcal{K}$ that are relevant to verifying the claim.
- **Label Prediction**: Assign a veracity label $y \in \{\text{SUPPORTS}, \text{REFUTES}, \text{NOT ENOUGH INFO}\}$ grounded strictly upon the retrieved evidence $\mathcal{E}$.

Under benchmark rules, predicting the label correctly is insufficient: verifiable claims require that at least one complete annotated gold evidence set be present in the retrieved candidates.

---

## 3. Motivation and Objectives

- **Evidence Grounding**: Distinguish between claims refuted by verified evidence (`REFUTES`) and claims that simply lack sufficient evidence in the knowledge base (`NOT ENOUGH INFO`).
- **Core Information Retrieval Emphasis**: Demonstrate classic and probabilistic retrieval techniques (Inverted Indexes, Okapi BM25, TF-IDF Cosine similarity, Boolean logic, Reciprocal Rank Fusion) at sentence granularity.
- **Explainability**: Provide human-interpretable reasoning showing exactly which sentence was retrieved, its ranking position, score, and why the verdict was rendered.
- **Reproducibility**: Run completely locally with zero proprietary API keys, deterministic seeds, modular code architecture, and a test suite.

---

## 4. Research Questions

- **RQ1**: How do classical lexical and probabilistic retrieval models (Okapi BM25, TF-IDF, Boolean) compare in terms of sentence-level Mean Reciprocal Rank (MRR), Recall@K, and query latency?
- **RQ2**: How effectively can feature-engineered evidence signals (containment, Jaccard overlap, negation polarity, numeric discrepancy) distinguish true contradictions from evidence absence?
- **RQ3**: What is the performance discrepancy between unconstrained classification accuracy and the official joint FEVER score?

---

## 5. Scope & Operating Boundaries

- **In Scope**: Sentence-level Wikipedia inverted indexing, Okapi BM25 retrieval, TF-IDF vector space retrieval, Boolean queries, hybrid Reciprocal Rank Fusion, feature extraction, calibrated logistic regression classification, rule-based fallback, official FEVER benchmark scoring, Streamlit multi-page UI, SQLite caching.
- **Out of Scope / Future Work**: Cross-lingual claim verification, audio/image evidence, and full-web live search.

---

## 6. Dataset Description & Official Sources

The system uses the official **FEVER (Fact Extraction and VERification)** benchmark:
- **Paper**: Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification*. NAACL-HLT 2018.
- **Official Resources**: [https://fever.ai/resources.html](https://fever.ai/resources.html)
- **Data Splits**:
  - `train.jsonl`: 145,449 annotated claims with gold evidence sets.
  - `shared_task_dev.jsonl`: 19,998 evaluation claims.
  - `wiki-pages.zip`: 5.4 million Wikipedia articles in sentence-segmented lines format.
- **Sample Distribution (Included)**: Pre-packaged authentic benchmark sample in `data/sample/` (`train.jsonl`, `dev.jsonl`, `wiki_sample.jsonl`) for instant testing and faculty demonstrations without downloading 30 GB of dumps.

---

## 7. Software & Hardware Requirements

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Version 3.10 or 3.11.
- **RAM**: Minimum 4 GB for sample corpus; 16 GB for full Wikipedia dump.
- **Storage**: ~50 MB for core project and sample data.

---

## 8. System Architecture

```
+-----------------------------------------------------------------------------------+
|                              User Natural Language Claim                           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                           1. Text Preprocessing Pipeline                          |
|  - Unicode NFKD Normalization                                                     |
|  - Tokenization & Hyphenation Preservation                                        |
|  - Case-Folding + Named Entity & Date Extraction                                  |
|  - Negation Preservation (critical words: not, no, never retained)                |
|  - Porter Stemming                                                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     2. Inverted Index & Knowledge Base Search                     |
|  - Inverted Index Term Posting Traversal: [(record_id, term_frequency), ...]       |
|  - Selectable Models: Okapi BM25 / TF-IDF Cosine / Boolean / Hybrid RRF            |
|  - Ranked Top-K Candidate Sentences with Scores and Metadata                      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        3. Feature Engineering Engine                              |
|  - Max & Mean Retrieval Score + Score Margin                                      |
|  - Jaccard Similarity & Claim Token Containment                                   |
|  - Negation Polarity Mismatch Flag                                                |
|  - Numeric & Date Contradiction Ratio                                             |
|  - Entity Overlap & Token Length Ratio                                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                   4. Evidence-Grounded Verification & Reasoning                   |
|  - Multi-Class Calibrated Estimator (Logistic Regression / Random Forest)         |
|  - Rule-Grounded Fallback & Low-Evidence Abstention (NEI Threshold)               |
|  - Output: Verdict (SUPPORTS / REFUTES / NEI) + Grounded Explanatory Text        |
+-----------------------------------------------------------------------------------+
```

---

## 9. Key Algorithms Used

1. **Inverted Indexing**: Term-to-postings mappings with document frequency and collection length statistics.
2. **Okapi BM25**: Probabilistic term frequency saturation ($k_1 = 1.5$) and length normalization ($b = 0.75$) using non-negative Robertson-Spärck Jones IDF.
3. **TF-IDF & Cosine Similarity**: Sublinear TF scaling ($1 + \ln(\text{tf})$) over normalized $L_2$ document vectors.
4. **Reciprocal Rank Fusion (RRF)**: Combining heterogeneous rankers with rank discounting constant $k = 60$.
5. **Calibrated Classification**: Multinomial Logistic Regression with balanced class weighting.

---

## 10. Repository Structure

```
Information Retrieval System/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
├── configs/
│   └── config.yaml             # Centralized pipeline configuration
├── data/
│   ├── sample/                 # Verified sample FEVER benchmark files
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── wiki_sample.jsonl
│   ├── raw/                    # Directory for official large downloads
│   └── processed/
├── artifacts/
│   ├── indexes/                # Serialized index artifacts
│   ├── models/                 # Serialized model artifacts
│   └── results/                # Evaluation benchmark results
│       └── evaluation_results.json
├── src/
│   ├── data/                   # Data loading, validation, and downloaders
│   ├── preprocessing/          # Normalization, tokenization, stemming
│   ├── indexing/               # InvertedIndex and IndexManager
│   ├── retrieval/              # BM25, TF-IDF, Boolean, Hybrid, Semantic
│   ├── verification/           # Feature extractor, trainer, inference engine
│   ├── evaluation/             # Retrieval metrics, FEVER score, error analysis
│   ├── pipeline/               # End-to-end FactVerifierPipeline orchestrator
│   └── utils/                  # Config loader, storage resolver, SQLite DB
├── pages/
│   ├── 1_Fact_Verification.py   # Interactive verification page
│   ├── 2_IR_Explorer.py        # Knowledge base search and comparison
│   ├── 3_Dataset_Explorer.py   # FEVER claims and Wikipedia browser
│   ├── 4_Evaluation.py         # Performance charts and error analysis
│   └── 5_Methodology.py        # Mathematical formulas and viva prep
├── scripts/
│   ├── prepare_data.py         # Dataset preparation CLI
│   ├── build_index.py          # Knowledge base indexing CLI
│   ├── train_model.py          # Verification classifier training CLI
│   └── run_evaluation.py       # Full benchmark evaluation runner CLI
├── tests/                      # Pytest test suite (22 unit & integration tests)
└── docs/                       # Complete academic documentation
    ├── architecture.md
    ├── methodology.md
    ├── experiments.md
    ├── user_guide.md
    ├── limitations.md
    ├── academic_report_outline.md
    └── viva_questions.md
```

---

## 11. Setup & Installation Instructions

### Step 1: Clone and Navigate
```bash
cd "Information Retrieval System"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 12. Step-by-Step Execution Guide

### Step 1: Prepare Datasets
```bash
python scripts/prepare_data.py --mode sample
```

### Step 2: Build the Knowledge Base Index
```bash
python scripts/build_index.py --corpus data/sample/wiki_sample.jsonl
```

### Step 3: Train the Verification Classifier
```bash
python scripts/train_model.py --train_file data/sample/train.jsonl
```

### Step 4: Run Evaluation Benchmark
```bash
python scripts/run_evaluation.py --dev_file data/sample/dev.jsonl
```

### Step 5: Execute Unit & Integration Tests
```bash
python -m pytest -p no:cacheprovider -v
```

### Step 6: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 13. Empirical Benchmark Results

Evaluated on the held-out development partition:

### Stage 1: Retrieval Performance
| Retrieval Method | MRR | Recall@5 | Precision@1 | nDCG@5 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Okapi)** | **0.8889** | **94.44%** | 77.78% | **0.8661** | **0.31 ms** |
| **Boolean** | 0.9444 | 94.44% | 88.89% | 0.9160 | 0.69 ms |
| **TF-IDF (Cosine)** | 0.8333 | 94.44% | 66.67% | 0.8409 | 1.43 ms |
| **Hybrid (RRF)** | 0.8889 | 94.44% | 77.78% | 0.8819 | 0.94 ms |

### Stage 2: Fact Verification Performance
| Benchmark Metric | Score | Note |
| :--- | :---: | :--- |
| **Official FEVER Score** | **83.33%** | Label correct **AND** complete gold evidence retrieved |
| **Label Accuracy** | **83.33%** | Label classification accuracy |
| **Evidence Recall** | **88.89%** | Complete gold evidence set present in top-$k$ |
| **Macro-F1** | **83.02%** | Balanced harmonic mean across all 3 classes |

---

## 14. Academic Documentation & Viva Preparation

Comprehensive academic documentation is located in `docs/`:
- [`docs/architecture.md`](docs/architecture.md): High-level system architecture and Mermaid diagrams.
- [`docs/methodology.md`](docs/methodology.md): Complete mathematical formulas for BM25, TF-IDF, RRF, and FEVER score.
- [`docs/experiments.md`](docs/experiments.md): Comprehensive benchmark results and confusion matrix analysis.
- [`docs/user_guide.md`](docs/user_guide.md): CLI operations guide and Streamlit tour.
- [`docs/limitations.md`](docs/limitations.md): Analysis of multi-hop, negation, and scalability challenges.
- [`docs/academic_report_outline.md`](docs/academic_report_outline.md): Complete academic project thesis outline.
- [`docs/viva_questions.md`](docs/viva_questions.md): 20+ viva voce examination questions with detailed model answers.

---

## 15. References & Citation

1. **Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A.** (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification*. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2018), pages 809–819.
2. **Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M., & Gatford, M.** (1994). *Okapi at TREC-3*. NIST Special Publication 500-225, pages 109–126.
3. **Cormack, G. V., Clarke, C. L., & Buettcher, S.** (2009). *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*. In Proceedings of the 32nd International ACM SIGIR Conference, pages 758–759.
4. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press.
