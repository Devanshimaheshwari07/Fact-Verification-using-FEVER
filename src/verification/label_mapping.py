"""Label mappings and normalization for FEVER verification classes."""
from typing import Dict

LABEL_TO_ID: Dict[str, int] = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

ID_TO_LABEL: Dict[int, str] = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

ALL_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def normalize_label(raw_label: str) -> str:
    """Normalizes label string to canonical FEVER form."""
    if not raw_label:
        return "NOT ENOUGH INFO"
    cleaned = raw_label.strip().upper()
    if cleaned in LABEL_TO_ID:
        return cleaned
    if cleaned.startswith("SUPP") or cleaned == "SUPPORT":
        return "SUPPORTS"
    if cleaned.startswith("REF") or cleaned == "REFUTE":
        return "REFUTES"
    return "NOT ENOUGH INFO"
