# Scope, Challenges & Known Limitations

This document provides a candid, academically rigorous analysis of the design trade-offs, scope boundaries, and known limitations of the FEVER Information Retrieval and Fact Verification System.

---

## 1. Information Retrieval Scope & Bottlenecks

### 1.1 Lexical vs. Semantic Gap
- **Limitation**: Pure lexical algorithms (BM25 and TF-IDF) rely on exact or stemmed term overlap. When a claim uses heavy paraphrasing, synonyms, or conceptual abstraction without overlapping vocabulary, lexical retrievers may fail to retrieve the relevant evidence sentence.
- **Mitigation & Trade-off**: The system incorporates `HybridRetriever` with Reciprocal Rank Fusion and dense semantic projection. However, dense transformer embeddings require significant computational overhead and memory.

### 1.2 Multi-Hop Evidence Aggregation
- **Limitation**: In standard FEVER, certain claims require combining evidence from two or more distinct sentences across different Wikipedia articles (e.g., verifying that person A and person B won a prize in the same year). Single-query sentence ranking can retrieve only one of the required sentences in top-$k$, failing the strict FEVER completeness criteria.
- **Future Direction**: Implementing iterative query reformulation where entities discovered in first-hop retrieved sentences generate secondary search queries.

---

## 2. Fact Verification Classification Limitations

### 2.1 Lexical Bias on NOT ENOUGH INFO
- **Limitation**: When a claim discusses real, prominent entities (e.g., "Francis Ford Coppola" and "The Godfather"), the retrieved sentences will have high lexical overlap and high BM25 relevance scores. A linear model can misinterpret this high entity topical overlap as confirmation of an unmentioned predicate.
- **Mitigation**: The system extracts explicit `claim_containment` and verifies whether numbers or dates match, setting an explicit confidence threshold to default to `NOT ENOUGH INFO`.

### 2.2 Numerical & Temporal Reasoning
- **Limitation**: While our system extracts numbers and dates to identify mismatches, it does not perform complex arithmetic reasoning (e.g., verifying that "died at age 42" matches "born in 1912 and died in 1954").

---

## 3. Knowledge Base & Hardware Constraints

### 3.1 Full Wikipedia Indexing Scale
- **Constraint**: The official FEVER Wikipedia dump contains over 5.4 million Wikipedia articles comprising more than 50 million sentences. Indexing the full dump requires approximately 16 GB of RAM and 30 GB of disk storage.
- **Architectural Solution**: The project provides a configurable, streaming index builder (`stream_wiki_sentences`) that processes pages in memory-conscious batches, enabling instant sample testing on ordinary laptops while supporting full-scale runs on research servers.

### 3.2 Operating System File Access Controls
- **Challenge Encountered**: On Windows systems with OneDrive and Controlled Folder Access active, user-level Python processes are restricted from writing to protected `Documents` folders.
- **Engineering Solution**: The system integrates an automated `StorageResolver` (`src/utils/config_loader.py`) that detects OS restrictions and redirects writable artifacts to `~/.fever_cache/`, preserving complete functionality without requiring administrative elevation.
