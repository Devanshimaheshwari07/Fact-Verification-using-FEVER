# Viva Voce Examination Questions & Model Answers

This guide prepares students and researchers for defense examinations and technical interviews on this Information Retrieval and Fact Verification project.

---

### Q1: What is the fundamental research problem addressed in automated fact verification?
**Answer:**  
Automated fact verification addresses the challenge of validating natural language claims against a trusted reference corpus (such as Wikipedia). Unlike simple question answering or document search, fact verification requires two interrelated steps:
1. Finding the minimal set of sentences that directly corroborate or refute the claim (Evidence Retrieval).
2. Inferring whether the claim logically follows from (`SUPPORTS`), contradicts (`REFUTES`), or cannot be proven by (`NOT ENOUGH INFO`) the retrieved evidence.

---

### Q2: Why is sentence-level retrieval more suitable than document-level retrieval for FEVER?
**Answer:**  
In fact verification, full Wikipedia articles often contain thousands of words across dozens of sub-topics. Passing an entire article to a verification model introduces substantial noise, irrelevant background context, and computational inefficiency. Sentence-level retrieval isolates the exact atomic propositions necessary to prove or disprove a claim, enabling transparent verification and rigorous evaluation against human-annotated evidence benchmarks.

---

### Q3: Explain the internal data structure of an Inverted Index and its time complexity advantage.
**Answer:**  
An Inverted Index consists of:
- **Vocabulary (Dictionary)**: A set of unique stemmed terms extracted from the corpus.
- **Postings Lists**: For each term $t$, an ordered list of tuples $[(\text{doc\_id}, \text{tf}_{t,d})]$ indicating which documents contain $t$ and how many times.
- **Document Metadata**: Document lengths and original strings.

**Time Complexity Advantage:**  
A linear scan checks all $N$ documents ($O(N \cdot L)$). With an inverted index, query processing only accesses the postings lists of query terms ($O(\sum_{t \in Q} |L_t|)$). Since individual terms appear in only a small fraction of the corpus, 99.9% of irrelevant documents are skipped without evaluation.

---

### Q4: State the Okapi BM25 formula and explain the physical meaning of $k_1$ and $b$.
**Answer:**  
$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- **$k_1$ (Term Frequency Saturation)**: In TF-IDF, score grows without bound as term frequency increases. In BM25, as $f(t, D) \to \infty$, the term factor approaches $k_1 + 1$. A value of $k_1 = 1.5$ means that the first occurrence of a term provides the largest gain, and subsequent occurrences provide diminishing marginal returns.
- **$b$ (Length Normalization)**: If $b = 1$, document length is fully normalized by $\frac{|D|}{\text{avgdl}}$, strongly penalizing long documents. If $b = 0$, length normalization is completely disabled. Setting $b = 0.75$ balances rewarding concise, focused sentences while not overly penalizing naturally detailed sentences.

---

### Q5: How does Robertson-Spärck Jones IDF differ from standard IDF?
**Answer:**  
Standard IDF is $\ln(N / n(t))$.  
Robertson-Spärck Jones IDF is:
$$\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1.0\right)$$
Adding $0.5$ provides continuity correction for small document counts, while adding $1.0$ ensures that even if a term appears in every document ($n(t) = N$), the IDF remains non-negative ($\ge 0$), preventing negative weights from corrupting scores.

---

### Q6: Why must stopword filtering be customized for fact verification?
**Answer:**  
In general ad-hoc web search, terms like `not`, `no`, `never`, `without` are frequently removed as high-frequency stopwords. In fact verification, negation completely reverses the semantic polarity and truth value of a claim (e.g. *"Alan Turing worked at Bletchley Park"* vs *"Alan Turing did not work at Bletchley Park"*). Discarding negation words would make contradictory claims identical to supported claims.

---

### Q7: What is the official FEVER Score and why is ordinary Accuracy insufficient?
**Answer:**  
Ordinary accuracy only measures whether the predicted label matches the ground truth label ($\hat{y} = y$). In fact verification, a model could memorize statistical dataset artifacts (e.g. claims with "never" are often REFUTES) and predict the right label without looking at evidence.  
The **official FEVER score** mandates that for `SUPPORTS` and `REFUTES`, a claim is only counted as correct if:
1. The predicted label is correct, **AND**
2. At least one complete gold evidence set is retrieved within the top-$k$ evidence list.  
For `NOT ENOUGH INFO`, since no evidence exists, predicting the label correctly suffices.

---

### Q8: Explain the difference between Precision@K, Recall@K, MRR, and nDCG@K.
**Answer:**  
- **Precision@K**: The proportion of the top-$k$ retrieved sentences that are relevant gold evidence: $|\mathcal{R}_k \cap \mathcal{E}| / k$.
- **Recall@K**: The fraction of all annotated gold evidence sentences retrieved in the top-$k$: $|\mathcal{R}_k \cap \mathcal{E}| / |\mathcal{E}|$.
- **Mean Reciprocal Rank (MRR)**: Evaluates ranking position; average of $1 / \text{rank}_1$ where $\text{rank}_1$ is the position of the first relevant sentence.
- **nDCG@K (Normalized Discounted Cumulative Gain)**: Measures graded relevance discounted logarithmically by rank position ($\sum \frac{\text{rel}_i}{\log_2(i+1)}$) divided by the Ideal DCG. It penalizes relevant results that appear lower down in the top-$k$ list.

---

### Q9: How does the system handle multi-hop evidence?
**Answer:**  
Multi-hop claims require synthesising information across multiple sentences. In our evaluation:
- We track multi-sentence evidence sets in `FeverClaim.evidence_sets`.
- Under FEVER evaluation rules, an evidence set is only marked as satisfied if *all* sentences in that set are present in the top-$k$ retrieved list.
- In `HybridRetriever`, we use Reciprocal Rank Fusion and entity-guided boosting to lift related sentences from the same or co-referenced Wikipedia articles into the candidate pool.

---

### Q10: How does the system distinguish REFUTES from NOT ENOUGH INFO?
**Answer:**  
- **NOT ENOUGH INFO**: Indicates the knowledge base does not contain sufficient information to prove or disprove the assertion. The system detects this through low overall retrieval scores, low token containment ($< 15\%$), and lack of entity grounding.
- **REFUTES**: Indicates the knowledge base *does* discuss the entities and predicates, but presents conflicting facts. The system detects this through high entity overlap coupled with explicit negation mismatches, numerical disagreements (e.g. year 1950 vs 1921), or antonymous predicates.

---

### Q11: What software engineering choices ensure reproducibility and deployment robustness?
**Answer:**  
1. **Zero External API Dependency**: Runs completely locally without third-party proprietary APIs or network requirements during inference.
2. **Deterministic Random Seeds**: Fixed random seed (42) for index building, dataset sampling, and model training.
3. **Storage Resilience**: Automated resolution between project directories and user application directories (`~/.fever_cache/`) to seamlessly handle OS-level permissions (e.g., Windows Defender Controlled Folder Access).
4. **Caching**: Streamlit `@st.cache_resource` and `@st.cache_data` ensure instant reloading without recomputing expensive indexes.
5. **Comprehensive Testing**: 22 unit and integration tests verifying every component from preprocessing to end-to-end pipeline predictions.
