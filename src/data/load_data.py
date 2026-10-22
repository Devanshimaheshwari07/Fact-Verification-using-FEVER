"""Data loading utilities for FEVER claims and Wikipedia knowledge base."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """Represents a single atomic piece of evidence in FEVER."""
    page_id: Optional[str]
    sentence_id: Optional[int]


@dataclass
class FeverClaim:
    """Structured representation of a FEVER claim."""
    claim_id: int
    claim: str
    label: str  # "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
    verifiable: str  # "VERIFIABLE" or "NOT VERIFIABLE"
    # List of alternative evidence sets. Each set is a list of EvidenceItems needed together.
    evidence_sets: List[List[EvidenceItem]] = field(default_factory=list)

    def get_flat_gold_evidence(self) -> Set[Tuple[str, int]]:
        """Returns set of all (page_id, sentence_id) pairs appearing in any gold evidence set."""
        gold = set()
        for ev_set in self.evidence_sets:
            for item in ev_set:
                if item.page_id is not None and item.sentence_id is not None:
                    gold.add((item.page_id, item.sentence_id))
        return gold


@dataclass
class EvidenceSentence:
    """Represents an indexed sentence from the Wikipedia knowledge base."""
    doc_id: str            # Wikipedia page ID / title (e.g., "Alan_Turing")
    sentence_id: int       # 0-indexed sentence ID within page
    text: str              # Original sentence text
    page_title: str        # Human readable page title (e.g., "Alan Turing")


def normalize_page_id(title: str) -> str:
    """Normalizes Wikipedia title: replaces spaces with underscores and unescapes symbols."""
    if not title:
        return ""
    return title.strip().replace(" ", "_")


def format_page_title(page_id: str) -> str:
    """Converts a Wikipedia page ID back into a human-readable title."""
    if not page_id:
        return ""
    return page_id.replace("_", " ")


def parse_fever_evidence(raw_evidence: Any) -> List[List[EvidenceItem]]:
    """Parses the nested FEVER evidence structure:

    evidence: [
      [ [annotation_id, evidence_id, page_id, sentence_id], ... ], # Evidence set 1
      [ [annotation_id, evidence_id, page_id, sentence_id], ... ]  # Alternative set 2
    ]
    """
    evidence_sets: List[List[EvidenceItem]] = []
    if not raw_evidence or not isinstance(raw_evidence, list):
        return evidence_sets

    for raw_set in raw_evidence:
        if not isinstance(raw_set, list):
            continue
        current_set: List[EvidenceItem] = []
        for item in raw_set:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                page_id = item[2]
                sentence_id = item[3]
                if page_id is not None and sentence_id is not None:
                    try:
                        sentence_id = int(sentence_id)
                        current_set.append(EvidenceItem(page_id=str(page_id), sentence_id=sentence_id))
                    except (ValueError, TypeError):
                        pass
        if current_set:
            evidence_sets.append(current_set)

    return evidence_sets


def load_fever_claims(file_path: str, max_records: Optional[int] = None) -> List[FeverClaim]:
    """Loads FEVER claims from a JSONL file with validation and error handling.

    Args:
        file_path: Path to the JSONL claims file.
        max_records: Maximum number of records to read (useful for testing).

    Returns:
        List of validated FeverClaim objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FEVER claims file not found: {path}")

    claims: List[FeverClaim] = []
    malformed_count = 0

    valid_labels = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Malformed JSON at line {line_no} in {path.name}: {e}")
                malformed_count += 1
                continue

            # Validate required fields
            if "claim" not in record or "id" not in record:
                logger.warning(f"Missing required fields at line {line_no} in {path.name}")
                malformed_count += 1
                continue

            claim_id = int(record["id"])
            claim_text = str(record["claim"]).strip()
            label = str(record.get("label", "NOT ENOUGH INFO")).strip().upper()
            if label not in valid_labels:
                logger.warning(f"Unknown label '{label}' at line {line_no}; defaulting to NOT ENOUGH INFO")
                label = "NOT ENOUGH INFO"

            verifiable = str(record.get("verifiable", "VERIFIABLE")).strip().upper()
            evidence_sets = parse_fever_evidence(record.get("evidence", []))

            claims.append(FeverClaim(
                claim_id=claim_id,
                claim=claim_text,
                label=label,
                verifiable=verifiable,
                evidence_sets=evidence_sets
            ))

            if max_records and len(claims) >= max_records:
                break

    logger.info(f"Loaded {len(claims)} valid claims from {path.name} ({malformed_count} malformed skipped).")
    return claims


def parse_wiki_lines_field(lines_str: str) -> List[Tuple[int, str]]:
    """Parses Wikipedia 'lines' field in official FEVER dump format.

    Format: '<sentence_id>\t<sentence_text>\t<entity_or_url_tags>\n...'
    """
    sentences = []
    if not lines_str:
        return sentences

    for raw_line in lines_str.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split("\t")
        if len(parts) >= 2:
            try:
                sent_id = int(parts[0])
                sent_text = parts[1].strip()
                if sent_text:
                    sentences.append((sent_id, sent_text))
            except (ValueError, TypeError):
                continue
    return sentences


def stream_wiki_sentences(file_path: str, max_pages: Optional[int] = None) -> Iterator[EvidenceSentence]:
    """Streams sentence-level records from a Wikipedia JSONL dump, directory of dumps, or sentence JSONL file."""
    path = Path(file_path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Wikipedia path not found: {path}")
        is_dir = path.is_dir()
    except (PermissionError, OSError):
        is_dir = False

    files: List[Path] = []
    if is_dir:
        try:
            files = sorted(list(path.glob("*.jsonl")))
            if not files:
                files = sorted(list(path.glob("*")))
        except (PermissionError, OSError):
            w1 = path / "wiki-001.jsonl"
            files = [w1] if w1.exists() else []
    else:
        files = [path]

    pages_read = 0
    for target_file in files:
        if not target_file.is_file() or target_file.suffix not in {".jsonl", ".txt", ".json"}:
            continue
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check if this is already an individual sentence record
                if "sentence_id" in record and "text" in record:
                    doc_id = str(record.get("doc_id", record.get("id", "")))
                    sent_id = int(record["sentence_id"])
                    text = str(record["text"]).strip()
                    if text:
                        yield EvidenceSentence(
                            doc_id=doc_id,
                            sentence_id=sent_id,
                            text=text,
                            page_title=format_page_title(doc_id)
                        )
                # Check if this is a FEVER wiki-pages dump record with 'lines'
                elif "id" in record and "lines" in record:
                    doc_id = str(record["id"])
                    page_title = format_page_title(doc_id)
                    sentences = parse_wiki_lines_field(record["lines"])
                    for sent_id, text in sentences:
                        yield EvidenceSentence(
                            doc_id=doc_id,
                            sentence_id=sent_id,
                            text=text,
                            page_title=page_title
                        )
                    pages_read += 1
                    if max_pages and pages_read >= max_pages:
                        return


def load_wiki_corpus(file_path: str, max_pages: Optional[int] = None) -> List[EvidenceSentence]:
    """Loads all sentences from a Wikipedia dump file into memory."""
    sentences = list(stream_wiki_sentences(file_path, max_pages=max_pages))
    logger.info(f"Loaded {len(sentences)} sentence records from {file_path}")
    return sentences
