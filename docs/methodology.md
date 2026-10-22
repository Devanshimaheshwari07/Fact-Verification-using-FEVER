# Methodology & Mathematical Formulations

This document provides the formal mathematical models and algorithmic methodologies employed throughout the FEVER Information Retrieval and Fact Verification pipeline.

---

## 1. Probabilistic Lexical Retrieval (Okapi BM25)

The Okapi BM25 scoring model scores document $D$ with respect to query $Q = \{q_1, q_2, \dots, q_n\}$:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### Parameters and Components:
- **Term Frequency $f(q_i, D)$**: Count of term $q_i$ within sentence $D$.
- **Document Length $|D|$**: Token length of sentence $D$.
- **Average Document Length $\text{avgdl}$**:
  $$\text{avgdl} = \frac{1}{N} \sum_{j=1}^N |D_j|$$
- **Term Frequency Saturation $k_1$**: Set to $1.5$. Controls how quickly term frequency contribution saturates.
- **Length Normalization $b$**: Set to $0.75$. Penalizes sentences longer than average, preventing verbosity bias.
- **Robertson-Spärck Jones IDF**:
  $$\text{IDF}(q_i) = \ln\left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1.0 \right)$$
  where $N$ is total indexed sentences and $n(q_i)$ is the number of sentences containing $q_i$. Adding $1.0$ guarantees strictly non-negative weights for all terms.

---

## 2. Vector Space Model (TF-IDF & Cosine Similarity)

### Sublinear Term Frequency Weighting:
To prevent repetitive terms from dominating linearly, sublinear TF scaling is applied:
$$\text{tf}_{t, D} = \begin{cases} 1 + \ln(f(t, D)) & \text{if } f(t, D) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Document Vector & Normalization:
$$\mathbf{d} = (w_{t_1, D}, w_{t_2, D}, \dots, w_{t_V, D})^T \quad \text{where } w_{t, D} = \text{tf}_{t, D} \cdot \text{idf}_t$$
Vectors are normalized to unit Euclidean length ($\|\mathbf{d}\|_2 = 1$). The cosine similarity between claim query vector $\mathbf{q}$ and sentence vector $\mathbf{d}$ reduces to the inner product:
$$\text{Cosine}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d} = \sum_{t \in Q \cap D} w_{t, Q} \cdot w_{t, D}$$

---

## 3. Two-Stage Hybrid Retrieval & Reciprocal Rank Fusion

In hybrid retrieval, evidence candidates are collected from both probabilistic ranking ($\mathcal{R}_{\text{BM25}}$) and vector space cosine similarity ($\mathcal{R}_{\text{TFIDF}}$). Scores are fused using Reciprocal Rank Fusion (RRF; Cormack et al., 2009):

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{TFIDF}\}} \frac{100.0}{k_{\text{rrf}} + \text{rank}_m(d)}$$

where $k_{\text{rrf}} = 60$ prevents top-ranked outliers from overshadowing consistent top-tier candidates across methods.

If named entities extracted from the claim appear in the Wikipedia page title of candidate sentence $d$, an entity boost multiplier ($\times 1.25$) is applied to prioritize topically grounded documents.

---

## 4. Feature Extraction & Verification Formulation

For a claim $C$ and top-$k$ retrieved evidence sentences $\mathcal{E} = \{e_1, e_2, \dots, e_k\}$, a 14-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{14}$ is constructed:

1. **Max Retrieval Score**: $\ln(1 + \max_{i} \text{score}(e_i))$
2. **Mean Retrieval Score**: $\ln(1 + \frac{1}{k} \sum_{i=1}^k \text{score}(e_i))$
3. **Score Margin**: $\ln(1 + (\text{score}(e_1) - \text{score}(e_2)))$
4. **Max Jaccard Overlap**: $\max_{i} \frac{|\text{Tokens}(C) \cap \text{Tokens}(e_i)|}{|\text{Tokens}(C) \cup \text{Tokens}(e_i)|}$
5. **Max Claim Containment**: $\max_{i} \frac{|\text{Tokens}(C) \cap \text{Tokens}(e_i)|}{|\text{Tokens}(C)|}$
6. **Claim Negation Flag**: $\mathbb{I}(\text{Negation} \in C)$
7. **Evidence Negation Flag**: $\mathbb{I}(\text{Negation} \in \mathcal{E})$
8. **Negation Mismatch Indicator**: $|\mathbb{I}(\text{Negation} \in C) - \mathbb{I}(\text{Negation} \in \mathcal{E})|$
9. **Claim Numbers Present**: $\mathbb{I}(|\text{Numbers}(C)| > 0)$
10. **Numeric Overlap Ratio**: $\frac{|\text{Numbers}(C) \cap \text{Numbers}(\mathcal{E})|}{|\text{Numbers}(C)|}$
11. **Numeric Mismatch Flag**: $\mathbb{I}(\text{Numbers}(C) \neq \emptyset \wedge |\text{Numbers}(C) \cap \text{Numbers}(\mathcal{E})| = 0)$
12. **Entity Overlap Ratio**: Fraction of extracted named entities found in evidence text or titles.
13. **Evidence Sentence Count**: $|\mathcal{E}|$
14. **Token Length Ratio**: $\min\left(\frac{|\text{Tokens}(C)|}{\max(|\text{Tokens}(e_1)|, 1)}, 3.0\right)$

### Classification Formulation:
A multinomial logistic regression classifier maps feature vector $\mathbf{x}$ to probability distribution $\mathbf{p} = (p_{\text{SUPPORTS}}, p_{\text{REFUTES}}, p_{\text{NEI}})^T$ via the softmax function:
$$p(y = j \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_j^T \mathbf{x} + b_j)}{\sum_{c=1}^3 \exp(\mathbf{w}_c^T \mathbf{x} + b_c)}$$

---

## 5. Official FEVER Benchmark Scoring Metric

Let $c$ be a claim with ground truth label $y_c \in \{\text{SUPPORTS}, \text{REFUTES}, \text{NOT ENOUGH INFO}\}$ and gold evidence sets $\mathcal{E}_c = \{S_1, S_2, \dots, S_m\}$. Let $\hat{y}_c$ be the predicted label and $\mathcal{R}_c^{(k)}$ be the top-$k$ retrieved sentence identifiers $(doc\_id, sent\_id)$.

1. **Label Accuracy**:
   $$\text{LabelAcc} = \frac{1}{|C|} \sum_{c \in C} \mathbb{I}(\hat{y}_c = y_c)$$

2. **Evidence Recall**:
   For verifiable claims ($y_c \neq \text{NEI}$):
   $$\text{EvRecall} = \frac{1}{|C_{\text{verifiable}}|} \sum_{c \in C_{\text{verifiable}}} \mathbb{I}\left( \exists S \in \mathcal{E}_c : S \subseteq \mathcal{R}_c^{(k)} \right)$$

3. **Official FEVER Score**:
   $$\text{FEVER\_Score} = \frac{1}{|C|} \sum_{c \in C} \mathbb{I}\left( \hat{y}_c = y_c \wedge (y_c = \text{NEI} \;\vee\; \exists S \in \mathcal{E}_c : S \subseteq \mathcal{R}_c^{(k)}) \right)$$
