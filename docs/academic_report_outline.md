# Academic Project Report Outline

**Title:** FEVER: An Evidence-Based Information Retrieval System for Automated Fact Verification  
**Author:** Academic Project Submission  
**Domain:** Information Retrieval, Natural Language Processing, Applied Machine Learning  

---

## Table of Contents

### 1. Abstract
- Summary of the automated fact-checking challenge.
- The dual retrieval-verification paradigm.
- Key empirical findings: BM25 achieving 0.8889 MRR, 94.44% Recall@5, and 83.33% official FEVER score on held-out development evaluation.

### 2. Introduction
- **2.1 Motivation**: Spread of digital misinformation and necessity of automated, evidence-grounded verification.
- **2.2 Problem Statement**: Given claim $C$ and knowledge base $\mathcal{K}$, retrieve evidence $E \subseteq \mathcal{K}$ and predict $y \in \{\text{SUPPORTS}, \text{REFUTES}, \text{NOT ENOUGH INFO}\}$.
- **2.3 Research Questions (RQs)**:
  - *RQ1*: How do classical IR models (BM25, TF-IDF, Boolean) compare in sentence-level evidence retrieval latency and recall?
  - *RQ2*: To what extent does feature-engineered evidence grounding distinguish contradiction (`REFUTES`) from evidence absence (`NOT ENOUGH INFO`)?
  - *RQ3*: How does the official joint FEVER score compare against unconstrained label accuracy?
- **2.4 Contributions**: End-to-end reproducible open-source pipeline, transparent feature attribution, multi-model evaluation dashboard.

### 3. Literature Review
- **3.1 Fact Verification Benchmarks**: The FEVER shared task (Thorne et al., NAACL 2018).
- **3.2 Classical Information Retrieval**: Vector Space Models (Salton et al.), Probabilistic Relevance Framework & Okapi BM25 (Robertson et al., TREC-3).
- **3.3 Evidence-Based NLP & NLI**: Multi-hop reasoning and entailment estimation.

### 4. System Architecture & Methodology
- **4.1 Text Normalization Pipeline**: Unicode NFKD decomposition, Porter stemming, entity and date extraction.
- **4.2 Inverted Index & Postings**: Data structure design, frequency accumulation, collection statistics.
- **4.3 Retrieval Models**:
  - Exact Boolean retrieval (Intersection, Union, Difference).
  - Sublinear TF-IDF Vector Space Model with Cosine Similarity.
  - Okapi BM25 probabilistic model ($k_1=1.5, b=0.75$).
  - Hybrid Reciprocal Rank Fusion (RRF) with entity title prior.
- **4.4 Evidence Selection & Feature Engineering**:
  - Overlap, containment, negation mismatch detection, numeric inconsistency ratio.
- **4.5 Verification Classification**:
  - Multi-class Logistic Regression with class-weight calibration.
  - Rule-grounded fallback and low-evidence abstention logic.

### 5. Experimental Setup & Evaluation
- **5.1 Dataset Description**: Official FEVER train and development partitions.
- **5.2 Metrics Defined**:
  - IR Metrics: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), nDCG@K, Latency.
  - Verification Metrics: Per-class Precision/Recall/F1, Macro-F1, Confusion Matrix.
  - Benchmark Score: Official FEVER Score (joint label correctness and evidence completeness).
- **5.3 Comparative Retrieval Results Table**.
- **5.4 Classification Performance Breakdown**.

### 6. Error Analysis & Discussion
- **6.1 Retrieval Failures**: Missing multi-hop evidence sentences.
- **6.2 Polarity Confusion**: Subtlety of implicit negation without explicit negation keywords.
- **6.3 Spurious Predictions on NEI**: Entity topical overlap mistaken for predicate verification.

### 7. Software Architecture & Implementation
- **7.1 Modular Package Organization (`src/`)**.
- **7.2 Resilient Storage Resolver for Restricted OS Environments**.
- **7.3 Interactive Streamlit Dashboard**.
- **7.4 Comprehensive Unit & Integration Test Suite (`pytest`)**.

### 8. Conclusion & Future Work
- Summary of achievements.
- Scaling to dense neural cross-encoders and iterative query expansion.

### 9. References
- Thorne et al. (2018), "FEVER: a large-scale dataset for Fact Extraction and VERification", NAACL-HLT.
- Robertson et al. (1994), "Okapi at TREC-3", NIST Special Publication.
- Cormack et al. (2009), "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods", SIGIR.
- Manning et al. (2008), *Introduction to Information Retrieval*, Cambridge University Press.
