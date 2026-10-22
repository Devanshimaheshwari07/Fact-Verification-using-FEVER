"""Live demonstration CLI script showing end-to-end evidence retrieval and fact verification."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.fact_verifier import FactVerifierPipeline

def run_demo():
    print("=" * 80)
    print("FEVER INFORMATION RETRIEVAL & FACT VERIFICATION SYSTEM - LIVE RUN")
    print("=" * 80)

    pipeline = FactVerifierPipeline()

    test_claims = [
        ("Alan Turing was an English mathematician who broke German codes at Bletchley Park.", "SUPPORTS"),
        ("Buzz Aldrin was the first person to step on the Moon during Apollo 11.", "REFUTES"),
        ("Francis Ford Coppola refused to watch early test screenings of The Godfather.", "NOT ENOUGH INFO"),
        ("The first confirmed ascent of Mount Everest was achieved by Sir Edmund Hillary.", "SUPPORTS"),
        ("Albert Einstein received his Nobel Prize in Physics in the year 1950.", "REFUTES")
    ]

    for idx, (claim, expected) in enumerate(test_claims, 1):
        print(f"\n[{idx}] CLAIM: \"{claim}\"")
        print(f"    Expected Class: {expected}")
        print("    Running BM25 Retrieval & Verification...")

        output = pipeline.verify_claim(claim, retrieval_method="bm25", top_k=3, log_to_db=True)

        print(f"\n    >>> PREDICTED VERDICT: {output.predicted_label}")
        print(f"    >>> CONFIDENCE:        {output.confidence:.2%}")
        print(f"    >>> PROBABILITIES:     {output.probabilities}")
        print(f"\n    >>> RETRIEVED EVIDENCE GROUNDING:")
        for ev in output.evidence:
            print(f"        * Rank {ev.rank} [{ev.page_title} - Sent {ev.sentence_id}] (Score: {ev.score:.4f})")
            print(f"          \"{ev.text}\"")

        print(f"\n    >>> GROUNDED EXPLANATION:")
        print(f"        {output.explanation}")
        print("-" * 80)

if __name__ == "__main__":
    run_demo()
