# User & Operations Guide

## 1. Environment Setup

### Prerequisites
- Python 3.10 or 3.11 installed.
- Pip package manager.

### Installation
Clone or navigate to the repository directory:
```bash
cd "Information Retrieval System"
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Command-Line Pipeline Execution

The system can be operated entirely via standard modular command-line scripts:

### Stage 1: Prepare Datasets
Generate verified FEVER sample datasets for testing:
```bash
python scripts/prepare_data.py --mode sample
```
*Optional*: To download official multi-gigabyte benchmark splits:
```bash
python scripts/prepare_data.py --mode official --split dev
```

### Stage 2: Build the Knowledge Base Index
Construct and serialize the inverted index, document statistics, and TF-IDF representations:
```bash
python scripts/build_index.py --corpus data/sample/wiki_sample.jsonl
```

### Stage 3: Train Verification Classifier
Train the logistic regression verification classifier using features extracted from retrieved evidence:
```bash
python scripts/train_model.py --train_file data/sample/train.jsonl
```

### Stage 4: Run Evaluation Benchmark
Execute evaluation over held-out claims and output IR metrics, FEVER scores, and error analysis:
```bash
python scripts/run_evaluation.py --dev_file data/sample/dev.jsonl
```

### Stage 5: Run Test Suite
Execute the pytest suite:
```bash
python -m pytest -p no:cacheprovider -v
```

---

## 3. Launching the Interactive Streamlit Dashboard

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Navigating the Multi-Page Dashboard:
1. **Home / Project Overview (`app.py`)**: High-level status metrics, architecture flow, and quick claim verification widget.
2. **Fact Verification (`pages/1_Fact_Verification.py`)**: Detailed claim input, confidence distribution, evidence rank table, and grounded explanations.
3. **IR Explorer (`pages/2_IR_Explorer.py`)**: Search the knowledge base using Boolean, TF-IDF, BM25, or Hybrid retrieval with side-by-side rankings.
4. **Dataset Explorer (`pages/3_Dataset_Explorer.py`)**: Interactive claim table, class distributions, and Wikipedia sentence browser.
5. **Evaluation Dashboard (`pages/4_Evaluation.py`)**: Benchmark charts, confusion matrix, and error taxonomy case studies.
6. **Methodology (`pages/5_Methodology.py`)**: Algorithmic formulas, diagrams, and viva voce interview preparation.
