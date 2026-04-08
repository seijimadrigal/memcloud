#!/usr/bin/env python3
"""Add atomic facts to existing v7 DBs by extracting from raw engrams."""
import sys, os, json, re, time, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4.1-mini"
API_KEY = os.environ["OPENROUTER_API_KEY"]


def llm_call(prompt, max_tokens=1500):
    for attempt in range(3):
        try:
            resp = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_tokens},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def extract_atomic_facts(session_id, date, conv_text):
    prompt = f"""Extract ALL atomic facts from this conversation. Each fact should be a single, self-contained, searchable sentence.

RULES:
1. Each fact must express exactly ONE piece of information
2. Always state WHO — use full names, never pronouns
3. Include specific details: names, dates, places, titles, numbers
4. Resolve relative dates using session date ({date})
5. Filter out greetings and filler — only meaningful facts
6. Write in third person: "Emma likes blue" not "I like blue"
7. Split compound facts: "Emma likes blue and red" → two facts

Session Date: {date}

Conversation:
{conv_text}

Return a JSON array of objects, each with "subject" (person name) and "fact" (the atomic fact sentence).
Return ONLY the JSON array."""

    result = llm_call(prompt)
    json_match = re.search(r'\[.*\]', result, re.DOTALL)
    if json_match:
        facts = json.loads(json_match.group())
        return [f for f in facts if isinstance(f, dict) and "fact" in f]
    return []


def process_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create atomic_facts tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS atomic_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            date TEXT,
            fact_text TEXT NOT NULL,
            subject TEXT,
            created_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_atomic_session ON atomic_facts(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_atomic_subject ON atomic_facts(subject)")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS atomic_facts_fts USING fts5(
            fact_text, subject, session_id,
            tokenize='porter unicode61'
        )
    """)
    
    # Check if already populated
    existing = conn.execute("SELECT COUNT(*) as cnt FROM atomic_facts").fetchone()["cnt"]
    if existing > 0:
        print(f"  Already has {existing} atomic facts, skipping")
        conn.close()
        return

    # Get all engrams
    engrams = conn.execute("SELECT session_id, date, raw_text FROM engrams ORDER BY date").fetchall()
    print(f"  Processing {len(engrams)} sessions...")
    
    total_facts = 0
    for eng in engrams:
        session_id = eng["session_id"]
        date = eng["date"]
        raw_text = eng["raw_text"]
        
        try:
            facts = extract_atomic_facts(session_id, date, raw_text)
            for f in facts:
                fact_text = f.get("fact", "")
                subject = f.get("subject", "")
                if not fact_text:
                    continue
                conn.execute(
                    "INSERT INTO atomic_facts (session_id, date, fact_text, subject, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                    (session_id, date, fact_text, subject),
                )
                conn.execute(
                    "INSERT INTO atomic_facts_fts (fact_text, subject, session_id) VALUES (?, ?, ?)",
                    (fact_text, subject, session_id),
                )
                total_facts += 1
            print(f"    {session_id}: {len(facts)} facts")
            time.sleep(0.3)
        except Exception as e:
            print(f"    {session_id}: ERROR {e}")
    
    conn.commit()
    conn.close()
    print(f"  Total: {total_facts} atomic facts")


if __name__ == "__main__":
    db_dir = sys.argv[1] if len(sys.argv) > 1 else "results/run12_v8"
    for db_name in ["conv-26.db", "conv-30.db", "conv-44.db"]:
        db_path = os.path.join(db_dir, db_name)
        if os.path.exists(db_path):
            print(f"\nProcessing {db_name}...")
            process_db(db_path)
