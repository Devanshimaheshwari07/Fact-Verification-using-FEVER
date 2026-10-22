"""SQLite database layer for caching predictions, queries, and experiment runs."""
import os
import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from src.utils.config_loader import get_storage_dir


class ExperimentDatabase:
    """Manages SQLite tables for logging predictions, user claim evaluations, and benchmark results."""

    def __init__(self, db_path: str = "artifacts/results/fever_ir.db"):
        results_dir = get_storage_dir("artifacts/results", for_writing=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(results_dir / Path(db_path).name)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for real-time verification queries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS claim_verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_text TEXT NOT NULL,
                    retrieval_method TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    explanation TEXT,
                    evidence_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table for offline evaluation metrics runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT NOT NULL,
                    dataset_split TEXT NOT NULL,
                    retrieval_method TEXT NOT NULL,
                    classifier_type TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def log_claim_verification(
        self,
        claim_text: str,
        retrieval_method: str,
        top_k: int,
        predicted_label: str,
        confidence: float,
        explanation: str,
        evidence: List[Dict[str, Any]]
    ) -> int:
        """Stores a user claim verification run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO claim_verifications (
                    claim_text, retrieval_method, top_k, predicted_label,
                    confidence, explanation, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                claim_text,
                retrieval_method,
                top_k,
                predicted_label,
                confidence,
                explanation,
                json.dumps(evidence)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_verifications(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recently executed claim verifications."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM claim_verifications
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["evidence"] = json.loads(item["evidence_json"])
                results.append(item)
            return results

    def log_evaluation_run(
        self,
        run_name: str,
        dataset_split: str,
        retrieval_method: str,
        classifier_type: str,
        num_samples: int,
        metrics: Dict[str, Any]
    ) -> int:
        """Logs an evaluation experiment result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluation_runs (
                    run_name, dataset_split, retrieval_method, classifier_type,
                    num_samples, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_name,
                dataset_split,
                retrieval_method,
                classifier_type,
                num_samples,
                json.dumps(metrics)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_evaluation_runs(self) -> List[Dict[str, Any]]:
        """Returns all recorded evaluation experiment runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM evaluation_runs
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["metrics"] = json.loads(item["metrics_json"])
                results.append(item)
            return results
