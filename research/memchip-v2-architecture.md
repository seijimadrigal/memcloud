# MemChip v2: "Cognitive Mesh" Architecture

## The Innovation: Multi-Resolution Memory with Adaptive Recall

Inspired by Complementary Learning Systems theory (McClelland et al., 1995) — the neuroscience of how the hippocampus and neocortex work together. No competitor does this.

### The Key Insight Nobody's Exploiting

**Human memory doesn't store everything at one resolution.** You remember:
- **Who your best friend is** → instant, no effort (semantic/gist memory)
- **What you talked about last Tuesday** → takes a moment, reconstructive (episodic memory)  
- **The exact words they said** → requires effort, often fails (verbatim memory)

Every competitor stores ONE type of memory and retrieves from it. We store THREE resolutions and route queries to the right one(s).

---

## Architecture: Three Memory Layers

### Layer 1: "Cortex" — Semantic Knowledge (Always Hot)
**What:** Compressed, structured knowledge. Entity profiles, relationships, key facts.
**Size:** ~500 tokens per conversation (tiny)
**Cost:** Near-zero to query
**Format:** A living knowledge document per entity, auto-updated after each session

Example:
```
## Caroline
- Identity: Transgender woman, she/her
- Age: ~28 (18th birthday was ~10 years ago)
- Interests: Art (painting, watercolors, sunset scenes), LGBTQ advocacy
- Relationships: Close friend of Melanie, has a cat named Whiskers
- Career: Works at an art gallery, considering art therapy
- Key events: Attended LGBTQ conference (July 2023), gave speech at school
```

**How it's built:** After each session, an LLM reads the session + existing profile and UPDATES the profile. This is consolidation — like sleep for the brain.

### Layer 2: "Hippocampus" — Episode Memory (Session-Level)
**What:** One structured summary per session, with date, key events, and emotional tone.
**Size:** ~100-200 tokens per session, ~2000-3000 for 19 sessions
**Cost:** Cheap to query
**Format:**

```
Session 7 | July 12, 2023 | Caroline & Melanie
- Caroline showed Melanie her new sunset watercolor painting
- Melanie mentioned signing up for a pottery class (July 2)
- They discussed the upcoming transgender conference
- Melanie's kids went to science camp last week (~July 5)
- Emotional tone: Excited, supportive
```

**How it's built:** LLM summarizes each session at ingestion time. Crucially includes RESOLVED dates — "last week" is converted to "~July 5" at write time using the session date.

### Layer 3: "Engram" — Verbatim Traces (On-Demand)
**What:** Raw conversation text, stored by session, indexed by session_id.
**Size:** Full conversation (~16-26K tokens for LoCoMo)
**Cost:** Expensive per query, but only accessed when needed
**Format:** Raw text with session headers

**How it's stored:** Verbatim. No processing. Just stored with session_id + date metadata.

---

## The Magic: Adaptive Recall Router

When a query comes in, a lightweight classifier decides which layers to activate:

### Recall Strategies (chosen per query):

**Strategy A: "Gist Recall"** — Cortex only
- For: "What pet does Caroline have?" / "What are Melanie's hobbies?"
- Cost: ~500 tokens
- How: Read entity profiles, answer directly

**Strategy B: "Episode Recall"** — Cortex + Hippocampus
- For: "When did Melanie sign up for pottery?" / "What happened in July?"
- Cost: ~3000 tokens  
- How: Read profiles for context + scan episode summaries for temporal/event info

**Strategy C: "Deep Recall"** — Cortex + Hippocampus + targeted Engrams
- For: "What exactly did Caroline say about her art?" / Multi-hop questions
- Cost: ~5000-8000 tokens (only loads relevant sessions, not all)
- How: Use episode summaries to identify relevant sessions → load those verbatim → reason over them

**Strategy D: "Full Reconstruction"** — All layers, all sessions
- For: Complex multi-hop spanning many sessions, or when other strategies fail
- Cost: ~20K tokens (full conversation)
- How: Load everything, but structured: profiles at top → episode timeline → raw sessions
- Fallback when confidence is low

### The Confidence Loop (NOVEL)
After generating an answer, check confidence:
1. If the answer contains "I don't know" / "not mentioned" / uncertainty → **escalate to next strategy**
2. Strategy A → B → C → D
3. Most questions resolve at A or B (cheap). Hard questions auto-escalate (accurate).

This is **economically scalable**: easy questions cost 500 tokens, hard questions cost 20K. Average cost is ~2-3K tokens/query — 10x cheaper than full-context-every-time.

---

## Why This Beats Every Competitor

| System | Write-time intelligence | Read-time intelligence | Resolution | Scalable? |
|--------|------------------------|----------------------|------------|-----------|
| Mem0 | Extract facts | Vector search | Single | Yes but lossy |
| EverMemOS | MemCells + episodes | Agentic retrieval | Two-tier | Heavy infra |
| Honcho | Fine-tuned ingestion + dreaming | Research agent | Single | Needs fine-tuning |
| Hindsight | 4 networks | Multi-strategy search | Single per network | Complex |
| Zep | Temporal KG | Graph traversal | Single | Heavy (Neo4j) |
| **MemChip v2** | **3-resolution consolidation** | **Adaptive routing + confidence escalation** | **Multi-resolution** | **SQLite, lightweight** |

### Our unique advantages:
1. **Adaptive cost**: 500 tokens for easy queries, 20K for hard ones. Nobody else does this.
2. **Zero information loss**: Raw text always available as fallback. Extraction errors can't kill us.
3. **Temporal resolution at write time**: "Last week" → "~July 5, 2023" baked into episode summaries. No ambiguity at query time.
4. **Confidence-based escalation**: Novel mechanism. If gist recall fails, automatically deepens. Like a human going "wait, let me think harder about that..."
5. **SQLite-only**: No MongoDB, no Neo4j, no Elasticsearch. One file. Runs everywhere.
6. **Pre-computed entity profiles**: Multi-hop questions about a person can often be answered from their profile alone — no graph traversal needed.

---

## Multi-Hop Innovation: "Decompose + Route"

For multi-hop questions like "What activities has Melanie done with her family?":

1. **Decompose** → "Who is in Melanie's family?" + "What activities has Melanie done?" + "Which of those involved family?"
2. **Route each sub-question** to the appropriate strategy:
   - "Who is in Melanie's family?" → Strategy A (Cortex: profile has this)
   - "What activities has Melanie done?" → Strategy B (Hippocampus: episode summaries list activities)
   - "Which involved family?" → Strategy C (Deep Recall: check specific sessions for family involvement)
3. **Synthesize** sub-answers into final answer

This is fundamentally different from competitors who either search everything or decompose but still search the same way for each sub-question.

---

## Temporal Innovation: "Resolved Timestamps"

The #1 failure mode in LoCoMo temporal questions: relative dates ("last week", "yesterday", "recently").

Our fix: **Resolve at WRITE time, not read time.**

When session_7 (July 12, 2023) mentions "I signed up for pottery class yesterday":
- Episode summary stores: "Melanie signed up for pottery class (~July 11, 2023)"
- Entity profile stores: "Pottery class (joined July 2023)"

When asked "When did Melanie sign up for pottery?":
- Strategy B reads episode summary → answer is right there
- No temporal reasoning needed at query time

Every competitor tries to resolve temporal references at query time. That's backwards. We do it at write time when we have full context.

---

## Implementation Plan

### Storage (SQLite, 3 tables):
```sql
-- Layer 1: Entity profiles (Cortex)
CREATE TABLE profiles (
    entity TEXT PRIMARY KEY,
    profile_text TEXT,  -- Markdown document
    updated_at TEXT
);

-- Layer 2: Episode summaries (Hippocampus)  
CREATE TABLE episodes (
    session_id TEXT PRIMARY KEY,
    date TEXT,           -- Absolute date
    date_iso TEXT,       -- ISO format for sorting
    summary TEXT,        -- Structured summary with resolved dates
    key_entities TEXT,   -- JSON array of entity names mentioned
    created_at TEXT
);

-- Layer 3: Raw sessions (Engram)
CREATE TABLE engrams (
    session_id TEXT PRIMARY KEY,
    date TEXT,
    raw_text TEXT,
    token_count INTEGER
);
```

Plus FTS5 index across all three layers for keyword search.

### Query Pipeline:
1. Classify query → identify entities + question type + estimated complexity
2. Select initial strategy (A/B/C/D)
3. Assemble context from selected layers
4. Generate answer with confidence indicator
5. If low confidence → escalate to deeper strategy
6. Return answer + metadata (which layers used, cost)

### Benchmark Approach:
- Run on LoCoMo with adaptive routing
- Track per-question: strategy used, tokens consumed, accuracy
- Optimize: which questions need which strategy
- Target: 93%+ accuracy at ~3K avg tokens/query
