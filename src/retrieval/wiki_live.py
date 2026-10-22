"""Dynamic Wikipedia retrieval and real-time knowledge base expansion."""
import json
import logging
import math
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Set, Tuple, Any

from src.data.load_data import EvidenceSentence, format_page_title, normalize_page_id

logger = logging.getLogger(__name__)

USER_AGENT = "FeverFactVerifier/1.0 (Academic IR Research Project; python-urllib)"


def search_wikipedia_titles(query: str, limit: int = 3) -> List[str]:
    """Queries Wikipedia's OpenSearch API for relevant article titles.

    Args:
        query: Search keywords or entity text.
        limit: Max number of article titles to return.

    Returns:
        List of matching Wikipedia article titles.
    """
    if not query or not query.strip():
        return []

    params = {
        "action": "opensearch",
        "search": query.strip(),
        "limit": str(limit),
        "namespace": "0",
        "format": "json"
    }
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            if len(data) >= 2 and isinstance(data[1], list):
                return data[1]
    except Exception as e:
        logger.warning(f"Wikipedia title search failed for '{query}': {e}")

    return []


def search_wikipedia_fulltext(query: str, limit: int = 3) -> List[str]:
    """Queries Wikipedia's full-text search API to discover articles mentioning specific assertions.

    Args:
        query: Search keywords including subject and key attribute terms.
        limit: Max number of article titles to return.

    Returns:
        List of matching Wikipedia article titles.
    """
    if not query or not query.strip():
        return []

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": str(limit),
        "format": "json"
    }
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            search_items = data.get("query", {}).get("search", [])
            return [item["title"] for item in search_items if "title" in item]
    except Exception as e:
        logger.warning(f"Wikipedia full-text search failed for '{query}': {e}")

    return []


def fetch_wikipedia_sentences(
    title: str,
    max_sentences: int = 20,
    attribute_tokens: Optional[Set[str]] = None,
    preprocessor: Optional[Any] = None
) -> List[EvidenceSentence]:
    """Fetches plain text extract of a Wikipedia article and segments it into sentences.

    If attribute_tokens is specified, searches the entire article and prioritizes sentences
    that explicitly mention the key attribute (e.g. vegan, dietary habits, awards, etc.).

    Args:
        title: Wikipedia page title.
        max_sentences: Maximum sentences to return.
        attribute_tokens: Optional set of attribute tokens to prioritize from body text.
        preprocessor: Optional TextPreprocessor instance.

    Returns:
        List of EvidenceSentence objects with sentence IDs.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json",
        "redirects": "1"
    }
    # If no attribute tokens to search deep, restrict to intro for efficiency
    if not attribute_tokens:
        params["exintro"] = "1"

    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=7) as response:
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_info in pages.items():
                if page_id == "-1":
                    continue  # Page not found

                resolved_title = page_info.get("title", title)
                doc_id = normalize_page_id(resolved_title)
                extract = page_info.get("extract", "")
                if not extract:
                    continue

                # Segment text into sentences
                raw_sentences = re.split(r"(?<=[.!?])\s+", extract)
                cleaned_sentences: List[str] = []
                for s in raw_sentences:
                    s_clean = s.strip()
                    # Filter out short fragments, section headings (== Section ==), and formatting noise
                    if len(s_clean) > 20 and not s_clean.startswith("==") and not s_clean.endswith("=="):
                        # Strip any internal section headers if present
                        s_clean = re.sub(r"==+[^=]+==+", "", s_clean).strip()
                        if len(s_clean) > 20:
                            cleaned_sentences.append(s_clean)

                if not cleaned_sentences:
                    continue

                if preprocessor is None:
                    from src.preprocessing.text_preprocessor import TextPreprocessor
                    preprocessor = TextPreprocessor()

                # If attribute tokens provided, prioritize sentences matching attribute
                selected_sentences: List[Tuple[int, str]] = []
                seen_indices: Set[int] = set()

                if attribute_tokens:
                    for idx, s_text in enumerate(cleaned_sentences):
                        toks = set(preprocessor.preprocess(s_text))
                        if toks.intersection(attribute_tokens):
                            selected_sentences.append((idx, s_text))
                            seen_indices.add(idx)

                # Always include top introductory sentences (first 4 sentences) for context
                for idx in range(min(4, len(cleaned_sentences))):
                    if idx not in seen_indices:
                        selected_sentences.append((idx, cleaned_sentences[idx]))
                        seen_indices.add(idx)

                # Fill up to max_sentences with remaining sentences
                for idx, s_text in enumerate(cleaned_sentences):
                    if len(selected_sentences) >= max_sentences:
                        break
                    if idx not in seen_indices:
                        selected_sentences.append((idx, s_text))
                        seen_indices.add(idx)

                # Sort by original sentence index to preserve discourse flow
                selected_sentences.sort(key=lambda x: x[0])

                evidence_list: List[EvidenceSentence] = []
                for sent_idx, s_text in selected_sentences[:max_sentences]:
                    evidence_list.append(EvidenceSentence(
                        doc_id=doc_id,
                        sentence_id=sent_idx,
                        text=s_text,
                        page_title=resolved_title
                    ))

                return evidence_list

    except Exception as e:
        logger.warning(f"Failed fetching Wikipedia extract for '{title}': {e}")

    return []


def rank_wikipedia_sentences(
    query: str,
    sentences: List[EvidenceSentence],
    preprocessor: Optional[Any] = None,
    top_k: int = 5
) -> List[Any]:
    """Ranks extracted Wikipedia sentences using attribute-weighted BM25 lexical relevance."""
    from src.retrieval.base_retriever import SearchResult
    if not sentences or not query or not query.strip():
        return []

    if preprocessor is None:
        from src.preprocessing.text_preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()

    query_tokens = preprocessor.preprocess(query)
    if not query_tokens:
        return []

    # Decompose claim into subject and key attribute tokens
    components = preprocessor.extract_claim_components(query)
    attribute_tokens = components.get("attribute_tokens", set())
    subject = components.get("subject", "")
    subject_tokens = set(preprocessor.preprocess(subject)) if subject else set()
    norm_subject_id = normalize_page_id(subject).lower() if subject else ""

    doc_tokens_list = [preprocessor.preprocess(s.text) for s in sentences]
    doc_lengths = [len(toks) for toks in doc_tokens_list]
    N = len(sentences)
    avgdl = sum(doc_lengths) / N if N > 0 else 1.0

    k1 = 1.5
    b = 0.75

    df: Dict[str, int] = {}
    for toks in doc_tokens_list:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    scored_results: List[Tuple[float, EvidenceSentence]] = []
    for idx, s in enumerate(sentences):
        # Relevance sanity check: if sentence is from a third-party article, it must actually mention the subject
        is_subject_article = bool(norm_subject_id and (s.doc_id.lower() == norm_subject_id or norm_subject_id in s.doc_id.lower()))
        toks = doc_tokens_list[idx]
        has_subject_mention = any(st in toks for st in subject_tokens)
        if subject and not is_subject_article and not has_subject_mention:
            continue

        dl = doc_lengths[idx]
        tf_dict: Dict[str, int] = {}
        for t in toks:
            tf_dict[t] = tf_dict.get(t, 0) + 1

        score = 0.0
        attr_hits = 0
        for q_term in set(query_tokens):
            if q_term in tf_dict:
                term_df = df.get(q_term, 0)
                idf = math.log((N - term_df + 0.5) / (term_df + 0.5) + 1.0)
                tf = tf_dict[q_term]
                tf_component = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)))

                # Attribute term weighting: attribute matches are prioritized over mere entity mentions
                weight = 1.8 if q_term in attribute_tokens else 1.0
                if q_term in attribute_tokens:
                    attr_hits += 1

                score += weight * idf * tf_component

        # Add bonus if the sentence matches both entity and attribute
        if attr_hits > 0:
            score += 2.5 * attr_hits

        # Prefer sentences directly from the subject's own article
        if is_subject_article:
            score += 1.0

        if score > 0:
            scored_results.append((score, s))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    results: List[SearchResult] = []
    for rank, (score, s) in enumerate(scored_results[:top_k], start=1):
        results.append(SearchResult(
            record_id=900000 + rank,
            doc_id=s.doc_id,
            sentence_id=s.sentence_id,
            score=round(float(score), 4),
            rank=rank,
            text=s.text,
            page_title=s.page_title or s.doc_id.replace("_", " "),
            retrieval_method="live_bm25"
        ))

    return results


def live_search(
    query: str,
    preprocessor: Optional[Any] = None,
    top_k: int = 5,
    max_articles: int = 3
) -> List[Any]:
    """Performs live end-to-end Wikipedia retrieval and BM25 ranking for any user claim or query.

    Decomposes the query into subject entities and asserted attributes, conducts dual
    OpenSearch and full-text Wikipedia discovery, fetches candidate sentences across
    lead and body sections, and applies attribute-weighted BM25 ranking.
    """
    if not query or not query.strip():
        return []

    if preprocessor is None:
        from src.preprocessing.text_preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()

    components = preprocessor.extract_claim_components(query)
    subject = components.get("subject", "")
    attr_text = components.get("attribute_text", "")
    attr_tokens = components.get("attribute_tokens", set())

    candidate_titles: List[str] = []
    seen_articles: Set[str] = set()

    def add_titles(titles: List[str]):
        for t in titles:
            if t and t not in seen_articles and len(seen_articles) < max_articles * 2:
                seen_articles.add(t)
                candidate_titles.append(t)

    # 1. Entity discovery: Search for primary subject article first
    if subject:
        title_matches = search_wikipedia_titles(subject, limit=2)
        add_titles(title_matches)

    # 2. Targeted discovery: Search for subject + attribute in full text
    if subject and attr_text:
        fulltext_titles = search_wikipedia_fulltext(f'"{subject}" {attr_text}', limit=2)
        add_titles(fulltext_titles)

    # 3. Fallback: Search full query
    if not candidate_titles:
        fallback_titles = search_wikipedia_titles(query, limit=2)
        add_titles(fallback_titles)
        if not candidate_titles and attr_text:
            add_titles(search_wikipedia_fulltext(query, limit=2))

    all_sentences: List[EvidenceSentence] = []

    for title in candidate_titles[:max_articles]:
        sents = fetch_wikipedia_sentences(
            title=title,
            max_sentences=25,
            attribute_tokens=attr_tokens,
            preprocessor=preprocessor
        )
        all_sentences.extend(sents)

    return rank_wikipedia_sentences(
        query=query,
        sentences=all_sentences,
        preprocessor=preprocessor,
        top_k=top_k
    )
