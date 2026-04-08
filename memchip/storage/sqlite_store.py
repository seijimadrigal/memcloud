"""
SQLite storage with FTS5 full-text search + vector embeddings + knowledge graph.
Local-first, zero infrastructure, EU AI Act compliant by default.
"""

from __future__ import annotations

import sqlite3
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


class SQLiteStore:
    def __init__(self, db_path: str = "memchip.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables for all memory types."""
        c = self.conn.cursor()

        # Semantic triples (atomic facts)
        c.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp TEXT,
                superseded_by INTEGER,
                embedding BLOB,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Conversation summaries
        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                summary TEXT NOT NULL,
                timestamp TEXT,
                embedding BLOB,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Entities (knowledge graph nodes)
        c.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                aliases TEXT,
                created_at REAL DEFAULT (strftime('%s','now')),
                UNIQUE(user_id, name)
            )
        """)

        # Entity relations (knowledge graph edges)
        c.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source_entity TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                session_id TEXT,
                timestamp TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Temporal events
        c.execute("""
            CREATE TABLE IF NOT EXISTS temporal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event TEXT NOT NULL,
                timestamp_raw TEXT,
                absolute_date TEXT,
                duration TEXT,
                recurring INTEGER DEFAULT 0,
                frequency TEXT,
                embedding BLOB,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Temporal ordering (before/after relations between events)
        c.execute("""
            CREATE TABLE IF NOT EXISTS temporal_order (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_before_id INTEGER,
                event_after_id INTEGER,
                relation TEXT DEFAULT 'before',
                FOREIGN KEY (event_before_id) REFERENCES temporal_events(id),
                FOREIGN KEY (event_after_id) REFERENCES temporal_events(id)
            )
        """)

        # Profile attributes
        c.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                person TEXT NOT NULL,
                category TEXT,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                superseded_by INTEGER,
                timestamp TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # FTS5 virtual table for full-text search across all text content
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content, memory_type, memory_id, user_id,
                tokenize='porter unicode61'
            )
        """)

        # Raw conversation text (for fallback search)
        c.execute("""
            CREATE TABLE IF NOT EXISTS raw_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                text TEXT NOT NULL,
                timestamp TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_triples_user ON triples(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_profiles_person ON profiles(person)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_temporal_user ON temporal_events(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_temporal_date ON temporal_events(absolute_date)")

        self.conn.commit()

    def store_extraction(
        self,
        extraction,
        user_id: str,
        session_id: str,
        timestamp: str,
    ) -> Dict[str, int]:
        """Store all extracted memories and index them for search."""
        c = self.conn.cursor()
        counts = {"triples": 0, "summaries": 0, "entities": 0, "temporal_events": 0, "profiles": 0}

        # Store triples
        for triple in extraction.triples:
            # Check for contradictions (same subject+predicate, different object)
            existing = c.execute(
                "SELECT id, object FROM triples WHERE user_id=? AND subject=? AND predicate=? AND superseded_by IS NULL",
                (user_id, triple.get("subject", ""), triple.get("predicate", ""))
            ).fetchone()

            if existing and existing["object"] != triple.get("object", ""):
                # Mark old triple as superseded
                c.execute("UPDATE triples SET superseded_by=? WHERE id=?", (-1, existing["id"]))

            try:
                conf = float(triple.get("confidence", 1.0))
            except (ValueError, TypeError):
                conf = 1.0
            c.execute(
                "INSERT INTO triples (user_id, session_id, subject, predicate, object, confidence, timestamp) VALUES (?,?,?,?,?,?,?)",
                (user_id, session_id, str(triple.get("subject", "")), str(triple.get("predicate", "")),
                 str(triple.get("object", "")), conf, timestamp)
            )
            triple_id = c.lastrowid
            # Index in FTS
            content = f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}"
            c.execute("INSERT INTO memory_fts (content, memory_type, memory_id, user_id) VALUES (?,?,?,?)",
                      (content, "triple", str(triple_id), user_id))
            counts["triples"] += 1

        # Store summary
        if extraction.summary:
            c.execute(
                "INSERT INTO summaries (user_id, session_id, summary, timestamp) VALUES (?,?,?,?)",
                (user_id, session_id, extraction.summary, timestamp)
            )
            summary_id = c.lastrowid
            c.execute("INSERT INTO memory_fts (content, memory_type, memory_id, user_id) VALUES (?,?,?,?)",
                      (extraction.summary, "summary", str(summary_id), user_id))
            counts["summaries"] = 1

        # Store entities
        for entity in extraction.entities:
            c.execute(
                "INSERT OR IGNORE INTO entities (user_id, name, entity_type, description, aliases) VALUES (?,?,?,?,?)",
                (user_id, str(entity.get("name", "")), str(entity.get("type", "")),
                 str(entity.get("description", "")), json.dumps(entity.get("aliases", [])))
            )
            counts["entities"] += 1

        # Store entity relations (derived from triples)
        for triple in extraction.triples:
            subj = triple.get("subject", "")
            obj = triple.get("object", "")
            pred = triple.get("predicate", "")
            if subj and obj:
                c.execute(
                    "INSERT INTO relations (user_id, source_entity, relation, target_entity, session_id, timestamp) VALUES (?,?,?,?,?,?)",
                    (user_id, subj, pred, obj, session_id, timestamp)
                )

        # Store temporal events
        for event in extraction.temporal_events:
            c.execute(
                "INSERT INTO temporal_events (user_id, event, timestamp_raw, absolute_date, duration, recurring, frequency) VALUES (?,?,?,?,?,?,?)",
                (user_id, str(event.get("event", "")), str(event.get("timestamp", "")),
                 str(event.get("absolute_date") or ""), str(event.get("duration") or ""),
                 1 if event.get("recurring") else 0, str(event.get("frequency") or ""))
            )
            event_id = c.lastrowid
            content = f"{event.get('event', '')} {event.get('timestamp', '')} {event.get('absolute_date', '')}"
            c.execute("INSERT INTO memory_fts (content, memory_type, memory_id, user_id) VALUES (?,?,?,?)",
                      (content, "temporal", str(event_id), user_id))
            counts["temporal_events"] += 1

        # Store profile attributes
        for attr in extraction.profile_attributes:
            # Check for contradictions
            existing = c.execute(
                "SELECT id, value FROM profiles WHERE user_id=? AND person=? AND attribute=? AND superseded_by IS NULL",
                (user_id, attr.get("person", ""), attr.get("attribute", ""))
            ).fetchone()

            if existing and existing["value"] != attr.get("value", ""):
                c.execute("UPDATE profiles SET superseded_by=? WHERE id=?", (-1, existing["id"]))

            try:
                pconf = float(attr.get("confidence", 1.0))
            except (ValueError, TypeError):
                pconf = 1.0
            c.execute(
                "INSERT INTO profiles (user_id, person, category, attribute, value, confidence, timestamp) VALUES (?,?,?,?,?,?,?)",
                (user_id, str(attr.get("person", "")), str(attr.get("category", "")),
                 str(attr.get("attribute", "")), str(attr.get("value", "")),
                 pconf, timestamp)
            )
            content = f"{attr.get('person', '')} {attr.get('attribute', '')} {attr.get('value', '')}"
            c.execute("INSERT INTO memory_fts (content, memory_type, memory_id, user_id) VALUES (?,?,?,?)",
                      (content, "profile", str(c.lastrowid), user_id))
            counts["profiles"] += 1

        self.conn.commit()
        return counts

    def store_raw(self, user_id: str, session_id: str, text: str, timestamp: str):
        """Store raw conversation text and index it in FTS."""
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO raw_sessions (user_id, session_id, text, timestamp) VALUES (?,?,?,?)",
            (user_id, session_id, text, timestamp)
        )
        raw_id = c.lastrowid
        c.execute("INSERT INTO memory_fts (content, memory_type, memory_id, user_id) VALUES (?,?,?,?)",
                  (text, "raw", str(raw_id), user_id))
        self.conn.commit()

    def search_raw(self, query: str, user_id: str, limit: int = 10) -> List[Dict]:
        """Search raw conversation text."""
        c = self.conn.cursor()
        # Use individual terms with OR for broad matching
        terms = [t for t in query.split() if len(t) > 2 and t.lower() not in {
            "what", "when", "where", "who", "how", "why", "did", "does", "the", "and",
            "was", "were", "are", "has", "have", "had", "that", "this", "with", "for",
            "from", "not", "but", "about", "which", "their", "they", "she", "his", "her",
        }]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms[:10])
        try:
            rows = c.execute(
                """SELECT text, session_id, timestamp FROM raw_sessions
                   WHERE user_id = ? AND rowid IN (
                       SELECT CAST(memory_id AS INTEGER) FROM memory_fts 
                       WHERE memory_fts MATCH ? AND user_id = ? AND memory_type = 'raw'
                       ORDER BY rank LIMIT ?
                   )""",
                (user_id, fts_query, user_id, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_session_dates(self, user_id: str) -> Dict[str, str]:
        """Get session_id → timestamp mapping."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT DISTINCT session_id, timestamp FROM summaries WHERE user_id = ? ORDER BY timestamp",
            (user_id,)
        ).fetchall()
        return {r["session_id"]: r["timestamp"] for r in rows if r["session_id"] and r["timestamp"]}

    def get_memory_timestamp(self, memory_type: str, memory_id: str) -> Optional[str]:
        """Get the timestamp for a memory by its type and id."""
        c = self.conn.cursor()
        table_map = {"triple": "triples", "summary": "summaries", "temporal": "temporal_events", "profile": "profiles"}
        table = table_map.get(memory_type)
        if not table:
            return None
        # temporal_events uses timestamp_raw/absolute_date instead of timestamp
        if table == "temporal_events":
            row = c.execute(f"SELECT absolute_date, timestamp_raw FROM {table} WHERE id = ?", (memory_id,)).fetchone()
            if row:
                return row["absolute_date"] or row["timestamp_raw"]
            return None
        try:
            row = c.execute(f"SELECT timestamp FROM {table} WHERE id = ?", (memory_id,)).fetchone()
            return row["timestamp"] if row else None
        except Exception:
            return None

    def search_fts(self, query: str, user_id: str, limit: int = 50) -> List[Dict]:
        """Full-text search using FTS5 (BM25 ranking)."""
        c = self.conn.cursor()
        results = []

        # First try: phrase match with escaped query
        safe_query = query.replace('"', '""')
        try:
            rows = c.execute(
                """SELECT content, memory_type, memory_id, rank 
                   FROM memory_fts 
                   WHERE memory_fts MATCH ? AND user_id = ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, user_id, limit)
            ).fetchall()
            results = [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass

        # Always also try individual terms with OR for broader matching
        if len(results) < limit:
            terms = [t for t in query.split() if len(t) > 2 and t.lower() not in {
                "what", "when", "where", "who", "how", "why", "did", "does", "the", "and",
                "was", "were", "are", "has", "have", "had", "that", "this", "with", "for",
                "from", "not", "but", "about", "which", "their", "they", "she", "his", "her",
            }]
            if terms:
                fts_query = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms[:10])
                try:
                    rows = c.execute(
                        """SELECT content, memory_type, memory_id, rank 
                           FROM memory_fts 
                           WHERE memory_fts MATCH ? AND user_id = ?
                           ORDER BY rank
                           LIMIT ?""",
                        (fts_query, user_id, limit)
                    ).fetchall()
                    existing = {r["content"] for r in results}
                    for r in rows:
                        d = dict(r)
                        if d["content"] not in existing:
                            results.append(d)
                            existing.add(d["content"])
                except sqlite3.OperationalError:
                    pass

        return results[:limit]

    def get_triples(self, user_id: str, subject: Optional[str] = None, active_only: bool = True) -> List[Dict]:
        """Get triples, optionally filtered by subject/object (case-insensitive)."""
        c = self.conn.cursor()
        query = "SELECT * FROM triples WHERE user_id = ?"
        params = [user_id]
        if active_only:
            query += " AND superseded_by IS NULL"
        if subject:
            query += " AND (subject LIKE ? COLLATE NOCASE OR object LIKE ? COLLATE NOCASE OR predicate LIKE ? COLLATE NOCASE)"
            params.extend([f"%{subject}%", f"%{subject}%", f"%{subject}%"])
        return [dict(r) for r in c.execute(query, params).fetchall()]

    def get_summaries(self, user_id: str) -> List[Dict]:
        """Get all summaries for a user."""
        c = self.conn.cursor()
        return [dict(r) for r in c.execute(
            "SELECT * FROM summaries WHERE user_id = ? ORDER BY timestamp", (user_id,)
        ).fetchall()]

    def get_profile(self, user_id: str, person: Optional[str] = None) -> List[Dict]:
        """Get profile attributes, optionally for a specific person or matching value."""
        c = self.conn.cursor()
        query = "SELECT * FROM profiles WHERE user_id = ? AND superseded_by IS NULL"
        params = [user_id]
        if person:
            query += " AND (person LIKE ? COLLATE NOCASE OR attribute LIKE ? COLLATE NOCASE OR value LIKE ? COLLATE NOCASE)"
            params.extend([f"%{person}%", f"%{person}%", f"%{person}%"])
        return [dict(r) for r in c.execute(query, params).fetchall()]

    def get_temporal_events(self, user_id: str, date_from: Optional[str] = None, date_to: Optional[str] = None) -> List[Dict]:
        """Get temporal events, optionally filtered by date range."""
        c = self.conn.cursor()
        query = "SELECT * FROM temporal_events WHERE user_id = ?"
        params = [user_id]
        if date_from:
            query += " AND absolute_date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND absolute_date <= ?"
            params.append(date_to)
        query += " ORDER BY absolute_date"
        return [dict(r) for r in c.execute(query, params).fetchall()]

    def graph_walk(self, user_id: str, entity: str, hops: int = 2) -> List[Dict]:
        """Walk the knowledge graph from an entity, up to N hops."""
        visited = set()
        results = []
        frontier = [entity]

        for hop in range(hops):
            next_frontier = []
            for ent in frontier:
                if ent in visited:
                    continue
                visited.add(ent)
                c = self.conn.cursor()
                # Find relations where entity is source or target
                rows = c.execute(
                    """SELECT source_entity, relation, target_entity, session_id, timestamp 
                       FROM relations WHERE user_id = ? AND (source_entity LIKE ? OR target_entity LIKE ?)""",
                    (user_id, f"%{ent}%", f"%{ent}%")
                ).fetchall()
                for row in rows:
                    r = dict(row)
                    r["hop"] = hop + 1
                    results.append(r)
                    # Add connected entities to next frontier
                    if r["source_entity"] not in visited:
                        next_frontier.append(r["source_entity"])
                    if r["target_entity"] not in visited:
                        next_frontier.append(r["target_entity"])
            frontier = next_frontier

        return results

    def clear(self, user_id: str):
        """Clear all memories for a user."""
        c = self.conn.cursor()
        for table in ["triples", "summaries", "entities", "relations",
                       "temporal_events", "temporal_order", "profiles", "raw_sessions"]:
            c.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM memory_fts WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
