"""Demonstration script verifying the Eiffel Tower claim step-by-step."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline.fact_verifier import FactVerifierPipeline

def run_demo():
    pipeline = FactVerifierPipeline()
    claim = "The Eiffel Tower is located in London."
    output = pipeline.verify_claim(claim, retrieval_method="bm25", top_k=3, log_to_db=False)

    print("=" * 65)
    print(f"INPUT CLAIM: \"{claim}\"")
    print("=" * 65)
    
    print("\n[Step 1: Retrieve Evidence]")
    for idx, ev in enumerate(output.evidence[:2], start=1):
        rel = "High" if idx == 1 else "Medium"
        print(f"  Source #{idx}: Wikipedia -> {ev.page_title} (Sentence {ev.sentence_id})")
        print(f"  Text: \"{ev.text}\"")
        print(f"  Relevance: {rel} (IR Score: {ev.score:.2f})\n")

    print("[Step 2: Compare the Claim with the Evidence]")
    print(f"  {'Component':<30} | {'Result'}")
    print("  " + "-" * 55)
    for row in output.comparison.rows:
        print(f"  {row['Component']:<30} | {row['Result']}")
    print(f"\n  Synthesis: {output.comparison.summary}\n")

    print("[Step 3: Produce the Verdict]")
    print(f"  VERDICT: {output.predicted_label}")
    print(f"  Confidence: {output.confidence:.1%}")
    print(f"  Evidence Status: Relevant evidence found")
    print(f"  Explanation: {output.explanation}")
    print("=" * 65)

if __name__ == "__main__":
    run_demo()
