#!/usr/bin/env python3
"""Copy DBs from v9.1 and add raw_chunks with v10.5 chunk sizes."""
import sqlite3, os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memchip.v10_5.core import chunk_text

SOURCE_DIR = "results/run16_v9.1"
DEST_DIR = "iterations/v10.5"
CONVS = ["conv-26", "conv-30", "conv-44"]

os.makedirs(DEST_DIR, exist_ok=True)

for conv in CONVS:
    src = os.path.join(SOURCE_DIR, f"{conv}.db")
    dst = os.path.join(DEST_DIR, f"{conv}.db")
    
    if not os.path.exists(src):
        print(f"SKIP {conv}: source not found")
        continue
    
    # Remove existing
    if os.path.exists(dst):
        os.remove(dst)
    
    # Copy
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")
    
    # Open and add raw_chunks
    db = sqlite3.connect(dst)
    db.row_factory = sqlite3.Row
    
    # Create raw_chunks table and FTS
    db.execute("""
        CREATE TABLE IF NOT EXISTS raw_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            date TEXT,
            UNIQUE(session_id, chunk_idx)
        )
    """)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS raw_chunks_fts USING fts5(
            text, content='raw_chunks', content_rowid='id',
            tokenize='porter unicode61'
        )
    """)
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS raw_chunks_ai AFTER INSERT ON raw_chunks BEGIN
            INSERT INTO raw_chunks_fts(rowid, text) VALUES (new.id, new.text);
        END
    """)
    db.commit()
    
    # Get all engrams and chunk them
    rows = db.execute("SELECT session_id, date, raw_text FROM engrams").fetchall()
    total_chunks = 0
    for r in rows:
        session_id = r["session_id"]
        date = r["date"]
        raw_text = r["raw_text"]
        chunks = chunk_text(raw_text)  # Uses new 150/75 params
        for i, chunk in enumerate(chunks):
            db.execute(
                "INSERT OR IGNORE INTO raw_chunks (session_id, chunk_idx, text, date) VALUES (?, ?, ?, ?)",
                (session_id, i, chunk, date),
            )
        total_chunks += len(chunks)
    
    db.commit()
    
    # Verify
    count = db.execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[0]
    fts_count = db.execute("SELECT COUNT(*) FROM raw_chunks_fts").fetchone()[0]
    print(f"  {conv}: {len(rows)} sessions -> {count} chunks ({fts_count} in FTS)")
    db.close()

print("\nDone! DBs ready for benchmark.")
