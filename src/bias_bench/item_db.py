"""
Item bank storage in SQLite with versioning.

Manages test items independently from the harness code.
Import/export between JSON (Pydantic models) and SQLite.
"""

import sqlite3
import json
from typing import Optional


def init_item_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS bank_versions (
            id INTEGER PRIMARY KEY,
            version TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS bias_families (
            id INTEGER PRIMARY KEY,
            family TEXT NOT NULL CHECK(family IN ('stereotype','framing','magnitude')),
            category TEXT NOT NULL CHECK(category IN ('optimization','human_hardware')),
            bank_version_id INTEGER REFERENCES bank_versions(id),
            UNIQUE(family, bank_version_id)
        );

        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            item_id TEXT NOT NULL,
            family_id INTEGER REFERENCES bias_families(id),
            family TEXT NOT NULL,
            item_type TEXT NOT NULL CHECK(item_type IN ('control','explicit','implicit')),
            prompt_text TEXT NOT NULL,
            correct_answer TEXT,
            metadata_json TEXT DEFAULT '{}',
            bank_version_id INTEGER REFERENCES bank_versions(id),
            UNIQUE(item_id, bank_version_id)
        );

        CREATE TABLE IF NOT EXISTS choices (
            id INTEGER PRIMARY KEY,
            item_pk INTEGER REFERENCES items(id),
            label TEXT NOT NULL,
            text TEXT NOT NULL,
            value REAL,
            sort_order INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS implicit_pairs (
            id INTEGER PRIMARY KEY,
            pair_id TEXT NOT NULL,
            family_id INTEGER REFERENCES bias_families(id),
            family TEXT NOT NULL,
            version_a_pk INTEGER REFERENCES items(id),
            version_b_pk INTEGER REFERENCES items(id),
            manipulation_description TEXT NOT NULL,
            expected_bias_direction TEXT NOT NULL,
            bank_version_id INTEGER REFERENCES bank_versions(id),
            UNIQUE(pair_id, bank_version_id)
        );

        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY,
            family_id INTEGER REFERENCES bias_families(id),
            control_pk INTEGER REFERENCES items(id),
            explicit_pk INTEGER REFERENCES items(id),
            implicit_pair_pk INTEGER REFERENCES implicit_pairs(id),
            bank_version_id INTEGER REFERENCES bank_versions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_items_version ON items(bank_version_id);
        CREATE INDEX IF NOT EXISTS idx_items_family ON items(family, item_type);
        CREATE INDEX IF NOT EXISTS idx_choices_item ON choices(item_pk);
        CREATE INDEX IF NOT EXISTS idx_pairs_version ON implicit_pairs(bank_version_id);
        CREATE INDEX IF NOT EXISTS idx_triples_version ON triples(bank_version_id);
    """)

    conn.commit()
    conn.close()


def _get_or_create_version(conn, version: str, description: str = None) -> int:
    c = conn.cursor()
    c.execute("SELECT id FROM bank_versions WHERE version = ?", (version,))
    row = c.fetchone()
    if row:
        return row[0]
    c.execute(
        "INSERT INTO bank_versions (version, description) VALUES (?, ?)",
        (version, description),
    )
    conn.commit()
    return c.lastrowid


def _get_or_create_family(conn, family: str, category: str, version_id: int) -> int:
    c = conn.cursor()
    c.execute(
        "SELECT id FROM bias_families WHERE family = ? AND bank_version_id = ?",
        (family, version_id),
    )
    row = c.fetchone()
    if row:
        return row[0]
    c.execute(
        "INSERT INTO bias_families (family, category, bank_version_id) VALUES (?, ?, ?)",
        (family, category, version_id),
    )
    conn.commit()
    return c.lastrowid


def _insert_item(conn, item: dict, family_id: int, family: str, version_id: int) -> int:
    """Insert a single item + its choices. Returns the item PK."""
    c = conn.cursor()

    # Skip if already exists
    c.execute(
        "SELECT id FROM items WHERE item_id = ? AND bank_version_id = ?",
        (item["id"], version_id),
    )
    existing = c.fetchone()
    if existing:
        return existing[0]

    metadata = json.dumps(item.get("metadata", {}))
    c.execute(
        """INSERT INTO items (item_id, family_id, family, item_type, prompt_text,
           correct_answer, metadata_json, bank_version_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item["id"], family_id, family, item["item_type"],
            item["prompt_text"], item.get("correct_answer"),
            metadata, version_id,
        ),
    )
    item_pk = c.lastrowid

    for i, ch in enumerate(item["choices"]):
        c.execute(
            "INSERT INTO choices (item_pk, label, text, value, sort_order) VALUES (?, ?, ?, ?, ?)",
            (item_pk, ch["label"], ch["text"], ch.get("value"), i),
        )

    return item_pk


def _insert_pair(conn, pair: dict, family_id: int, family: str, version_id: int) -> int:
    """Insert an implicit pair + its two items. Returns pair PK."""
    c = conn.cursor()

    c.execute(
        "SELECT id FROM implicit_pairs WHERE pair_id = ? AND bank_version_id = ?",
        (pair["id"], version_id),
    )
    existing = c.fetchone()
    if existing:
        return existing[0]

    va_pk = _insert_item(conn, pair["version_a"], family_id, family, version_id)
    vb_pk = _insert_item(conn, pair["version_b"], family_id, family, version_id)

    c.execute(
        """INSERT INTO implicit_pairs (pair_id, family_id, family, version_a_pk, version_b_pk,
           manipulation_description, expected_bias_direction, bank_version_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pair["id"], family_id, family, va_pk, vb_pk,
            pair["manipulation_description"], pair["expected_bias_direction"],
            version_id,
        ),
    )
    return c.lastrowid


def import_from_json(db_path: str, json_path: str, version: str) -> dict:
    """Import an ItemBank JSON file. Returns counts."""
    with open(json_path) as f:
        bank = json.load(f)

    conn = sqlite3.connect(db_path)
    version_id = _get_or_create_version(conn, version)
    counts = {"families": 0, "triples": 0, "items": 0}

    for fam in bank["families"]:
        family_id = _get_or_create_family(conn, fam["family"], fam["category"], version_id)
        counts["families"] += 1

        for triple in fam["triples"]:
            ctrl_pk = _insert_item(conn, triple["control"], family_id, fam["family"], version_id)
            expl_pk = _insert_item(conn, triple["explicit"], family_id, fam["family"], version_id)
            pair_pk = _insert_pair(conn, triple["implicit"], family_id, fam["family"], version_id)

            conn.cursor().execute(
                """INSERT INTO triples (family_id, control_pk, explicit_pk,
                   implicit_pair_pk, bank_version_id) VALUES (?, ?, ?, ?, ?)""",
                (family_id, ctrl_pk, expl_pk, pair_pk, version_id),
            )
            counts["triples"] += 1
            counts["items"] += 4  # control + explicit + 2 implicit versions

    conn.commit()
    conn.close()
    return counts


def import_triples_json(
    db_path: str, json_path: str, family: str, category: str, version: str
) -> dict:
    """Import a raw list-of-triples JSON (like stereotype chunk files)."""
    with open(json_path) as f:
        triples = json.load(f)

    conn = sqlite3.connect(db_path)
    version_id = _get_or_create_version(conn, version)
    family_id = _get_or_create_family(conn, family, category, version_id)
    count = 0

    for triple in triples:
        ctrl_pk = _insert_item(conn, triple["control"], family_id, family, version_id)
        expl_pk = _insert_item(conn, triple["explicit"], family_id, family, version_id)
        pair_pk = _insert_pair(conn, triple["implicit"], family_id, family, version_id)

        conn.cursor().execute(
            """INSERT INTO triples (family_id, control_pk, explicit_pk,
               implicit_pair_pk, bank_version_id) VALUES (?, ?, ?, ?, ?)""",
            (family_id, ctrl_pk, expl_pk, pair_pk, version_id),
        )
        count += 1

    conn.commit()
    conn.close()
    return {"triples": count, "items": count * 4}


def export_to_json(db_path: str, version: str) -> dict:
    """Export all items for a version as an ItemBank-compatible dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT id FROM bank_versions WHERE version = ?", (version,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Version '{version}' not found")
    version_id = row["id"]

    c.execute(
        "SELECT * FROM bias_families WHERE bank_version_id = ?", (version_id,)
    )
    families_rows = c.fetchall()

    families = []
    for fam_row in families_rows:
        fam_id = fam_row["id"]

        c.execute(
            "SELECT * FROM triples WHERE family_id = ? AND bank_version_id = ?",
            (fam_id, version_id),
        )
        triple_rows = c.fetchall()

        triples = []
        for tr in triple_rows:
            control = _load_item(conn, tr["control_pk"])
            explicit = _load_item(conn, tr["explicit_pk"])
            implicit = _load_pair(conn, tr["implicit_pair_pk"])
            triples.append({
                "control": control,
                "explicit": explicit,
                "implicit": implicit,
            })

        families.append({
            "family": fam_row["family"],
            "category": fam_row["category"],
            "triples": triples,
        })

    conn.close()
    return {"version": version, "families": families}


def _load_item(conn, item_pk: int) -> dict:
    c = conn.cursor()
    c.execute("SELECT * FROM items WHERE id = ?", (item_pk,))
    row = c.fetchone()

    c.execute("SELECT * FROM choices WHERE item_pk = ? ORDER BY sort_order", (item_pk,))
    choice_rows = c.fetchall()

    return {
        "id": row["item_id"],
        "family": row["family"],
        "item_type": row["item_type"],
        "prompt_text": row["prompt_text"],
        "correct_answer": row["correct_answer"],
        "metadata": json.loads(row["metadata_json"]),
        "choices": [
            {"label": ch["label"], "text": ch["text"], "value": ch["value"]}
            for ch in choice_rows
        ],
    }


def _load_pair(conn, pair_pk: int) -> dict:
    c = conn.cursor()
    c.execute("SELECT * FROM implicit_pairs WHERE id = ?", (pair_pk,))
    row = c.fetchone()

    return {
        "id": row["pair_id"],
        "family": row["family"],
        "version_a": _load_item(conn, row["version_a_pk"]),
        "version_b": _load_item(conn, row["version_b_pk"]),
        "manipulation_description": row["manipulation_description"],
        "expected_bias_direction": row["expected_bias_direction"],
    }


def get_families(db_path: str, version: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id FROM bank_versions WHERE version = ?", (version,))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    c.execute(
        "SELECT family, category FROM bias_families WHERE bank_version_id = ?",
        (row["id"],),
    )
    result = [dict(r) for r in c.fetchall()]
    conn.close()
    return result


def get_items(
    db_path: str, version: str, family: str = None, item_type: str = None
) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id FROM bank_versions WHERE version = ?", (version,))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    version_id = row["id"]

    query = "SELECT id FROM items WHERE bank_version_id = ?"
    params: list = [version_id]
    if family:
        query += " AND family = ?"
        params.append(family)
    if item_type:
        query += " AND item_type = ?"
        params.append(item_type)

    c.execute(query, params)
    items = [_load_item(conn, r["id"]) for r in c.fetchall()]
    conn.close()
    return items


def get_implicit_pairs(
    db_path: str, version: str, family: str = None
) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id FROM bank_versions WHERE version = ?", (version,))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    version_id = row["id"]

    query = "SELECT id FROM implicit_pairs WHERE bank_version_id = ?"
    params: list = [version_id]
    if family:
        query += " AND family = ?"
        params.append(family)

    c.execute(query, params)
    pairs = [_load_pair(conn, r["id"]) for r in c.fetchall()]
    conn.close()
    return pairs
