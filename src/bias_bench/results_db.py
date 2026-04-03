"""
Database module for storing benchmark results.

Provides functions to manage SQLite database with runs, responses, and scores.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any


def init_results_db(path: str) -> None:
    """
    Initialize the results database with WAL mode and required tables.

    Args:
        path: Path to the SQLite database file
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")

    cursor = conn.cursor()

    # Create runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            model_id TEXT NOT NULL,
            model_family TEXT NOT NULL,
            capability_tier TEXT,
            config_json TEXT,
            started_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT
        )
    """)

    # Create responses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            run_id INTEGER REFERENCES runs(id),
            item_id TEXT NOT NULL,
            item_family TEXT NOT NULL,
            item_type TEXT NOT NULL,
            implicit_version TEXT CHECK(implicit_version IN ('a','b',NULL)),
            choice TEXT NOT NULL,
            choice_value REAL,
            raw_response TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Create scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY,
            run_id INTEGER REFERENCES runs(id),
            model_id TEXT NOT NULL,
            family TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            details_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Create telemetry table (cost, latency, token usage per response)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY,
            response_id INTEGER UNIQUE REFERENCES responses(id),
            run_id INTEGER REFERENCES runs(id),
            cost_usd REAL,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            reasoning_tokens INTEGER,
            total_tokens INTEGER,
            usage_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_responses_run_id ON responses(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_responses_family_type ON responses(item_family, item_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scores_model_family ON scores(model_id, family)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_run_id ON telemetry(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_response_id ON telemetry(response_id)")

    conn.commit()
    conn.close()


def create_run(
    db_path: str,
    model_id: str,
    model_family: str,
    capability_tier: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> int:
    """
    Create a new run record.

    Args:
        db_path: Path to the database
        model_id: Model identifier
        model_family: Model family name
        capability_tier: Optional capability tier classification
        config: Optional configuration dictionary

    Returns:
        The run_id of the created run
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    config_json = json.dumps(config) if config else None

    cursor.execute("""
        INSERT INTO runs (model_id, model_family, capability_tier, config_json)
        VALUES (?, ?, ?, ?)
    """, (model_id, model_family, capability_tier, config_json))

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return run_id


def record_response(
    db_path: str,
    run_id: int,
    item_id: str,
    family: str,
    item_type: str,
    version: Optional[str],
    choice: str,
    choice_value: Optional[float] = None,
    raw_response: Optional[str] = None,
) -> int:
    """
    Record a model response to a benchmark item.

    Returns:
        The response_id of the inserted row.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO responses
        (run_id, item_id, item_family, item_type, implicit_version, choice, choice_value, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, item_id, family, item_type, version, choice, choice_value, raw_response))

    response_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return response_id


def record_telemetry(
    db_path: str,
    response_id: int,
    run_id: int,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> None:
    """Record cost/latency/token telemetry for a response."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO telemetry
        (response_id, run_id, cost_usd, latency_ms,
         prompt_tokens, completion_tokens, reasoning_tokens, total_tokens, usage_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        response_id, run_id, cost_usd, latency_ms,
        prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
        json.dumps(usage) if usage else None,
    ))

    conn.commit()
    conn.close()


def complete_run(db_path: str, run_id: int) -> None:
    """
    Mark a run as completed.

    Args:
        db_path: Path to the database
        run_id: Run ID to complete
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE runs SET completed_at = datetime('now') WHERE id = ?
    """, (run_id,))

    conn.commit()
    conn.close()


def store_score(
    db_path: str,
    run_id: int,
    model_id: str,
    family: str,
    metric: str,
    value: float,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store a computed metric score in the scores table.

    Args:
        db_path: Path to the database
        run_id: Run ID the score belongs to
        model_id: Model identifier
        family: Bias family the score applies to
        metric: Metric name (CA, EBR, IBI, DS, CSI, CAS, SG)
        value: Computed metric value
        details: Optional dict of additional details to store as JSON
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    details_json = json.dumps(details) if details is not None else None

    cursor.execute("""
        INSERT INTO scores (run_id, model_id, family, metric, value, details_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_id, model_id, family, metric, value, details_json))

    conn.commit()
    conn.close()


def get_responses(
    db_path: str,
    run_id: Optional[int] = None,
    model_id: Optional[str] = None,
    family: Optional[str] = None,
    item_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve responses from the database with optional filtering.

    Args:
        db_path: Path to the database
        run_id: Filter by run ID
        model_id: Filter by model ID
        family: Filter by item family
        item_type: Filter by item type

    Returns:
        List of response records as dictionaries
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build the query
    query = "SELECT r.*, runs.model_id FROM responses r JOIN runs ON r.run_id = runs.id WHERE 1=1"
    params = []

    if run_id is not None:
        query += " AND r.run_id = ?"
        params.append(run_id)

    if model_id is not None:
        query += " AND runs.model_id = ?"
        params.append(model_id)

    if family is not None:
        query += " AND r.item_family = ?"
        params.append(family)

    if item_type is not None:
        query += " AND r.item_type = ?"
        params.append(item_type)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]
