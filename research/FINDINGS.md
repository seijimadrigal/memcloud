# MemChip Research Findings — PERMANENT LOG

## ⚠️ CRITICAL: CATEGORY LABEL BUG (discovered 2026-04-01)

**LoCoMo dataset categories:** 1=multi-hop (282), 2=temporal (321), 3=open-domain (96), 4=single-hop (841), 5=adversarial (446)

**Our benchmark runner had 1↔4 SWAPPED for ALL runs v10-v24.** We labeled category 1 as "single-hop" and category 4 as "multi-hop". This means:
- What we reported as "single-hop 72%" was actually **multi-hop 72%**
- What we reported as "multi-hop 83%" was actually **single-hop 83%**
- We spent weeks optimizing single-hop (already at 83%) instead of multi-hop (actually 64-72%)

**TRUE scores (corrected):**
| Run | Single-hop (cat 4) | Multi-hop (cat 1) | Temporal | Overall |
|-----|-------------------|-------------------|----------|---------|
| v18 | **83%** | 64% | 90% | 77% |
| v17 | 81% | 68% | 92% | 79% |
| v10.4 | 79% | 72% | 95% | 81% |
| v24 | 77% | 38% | 89% | 72% |
| v23 | 70% | 31% | 84% | 66% |

**LESSON:** ALWAYS verify category/label mappings against the source dataset. Never trust inherited labels. Check the paper. This wasted 3+ days of optimization effort on the wrong target.

---

## DO NOT REPEAT THESE FAILED APPROACHES

### ❌ FAILED: Smaller Chunks (v10.5)
- Changed 250→150 words, overlap 50→75
- Result: 74.4% overall, single-hop 58.9% (was 71.9%)
- WHY: Splits facts across boundaries, loses context for reranker

### ❌ FAILED: FTS5 NEAR Queries (v10.6)
- Added NEAR(person, action, 15) co-occurrence search
- Result: 75.6% overall, single-hop 63.6%
- WHY: Too restrictive, misses chunks that OR query finds. Also adding atomic facts as supplementary noise didn't help.

### ❌ FAILED: Embedding Search / Hybrid BM25+Embedding+RRF (v12)
- Added all-MiniLM-L6-v2 embeddings, hybrid retrieval with RRF fusion
- Result: 78.3% overall, single-hop 67.1%
- WHY: Embedding results introduce noise that wasn't in FTS5-only results. The CrossEncoder reranker already handles semantic matching well enough on FTS5 candidates.

### ❌ FAILED: Gemini 2.5 Flash for Answering (v13)
- Swapped gpt-4.1-mini → Gemini 2.5 Flash for all LLM calls (incl. judge)
- Result: 58.2% overall, single-hop 34.9%
- WHY: 3 variables changed at once (ingestion + answering + judging). Gemini over-generates verbose answers. "Single parent" instead of "Single". Lists too many items.

### ❌ FAILED: Gemini Answering + GPT-4.1 Judge (v14)
- Same DBs as v10.2, Gemini answers, GPT-4.1 judges
- Result: 78.4% at 37q (killed early), single-hop 50%
- WHY: Gemini still over-generates. Model conciseness matters more than model intelligence for this benchmark format.

---

## ✅ BASELINE: v10.2 = 82.4% overall
| Category | Score |
|---|---|
| Single-hop | 71.9% |
| Temporal | 91.9% |
| Open-domain | 50.0% (worst) |
| Multi-hop | 82.9% |
| Adversarial | 83.0% |

### v10.2 Architecture:
- **Ingestion:** gpt-4.1-mini extracts profiles + episodes + atomic facts + temporal events + raw chunks (250 words, 50 overlap)
- **Single-hop retrieval:** FTS5 keyword search → CrossEncoder rerank (mxbai-rerank-large-v1) → top 4 chunks + entity profile
- **Single-hop answering:** gpt-4.1 (full, not mini) with precision prompt
- **Other categories:** strategy routing (A/B/C/D) with confidence escalation
- **Adversarial:** entity name masking
- **Judge:** gpt-4.1-mini

---

## SINGLE-HOP FAILURE PATTERNS (from v10.6 analysis, 24 failures)

### Pattern 1: Over-listing (model adds extra items)
- "Where camped?" → adds "Grand Canyon" (not in truth)
- "What to destress?" → adds "painting" (truth: running, pottery)
- "What painted recently?" → lists multiple (truth: just "sunset")
- **Root cause:** Profile/chunks contain more activities than ground truth expects. Model can't distinguish "important" from "mentioned once"

### Pattern 2: Wrong specific facts
- "How many children?" → says 2 (truth: 3). Facts say "two younger kids" + "son" + "daughter" = 3, but model counts wrong
- "Beach trips in 2023?" → says 1 (truth: 2). Needs to count across sessions
- "Both painted?" → says "self-portraits" (truth: "sunsets"). Wrong retrieval
- "City both visited?" → says "None" (truth: "Rome"). Missing from retrieved context

### Pattern 3: Vague answers
- "Family on hikes?" → generic "explore nature" (truth: "roast marshmallows, tell stories")
- "How promote store?" → wrong activities entirely
- **Root cause:** Profile summarization loses specific details. Raw chunks have specifics but aren't retrieved

---

## KEY INSIGHTS

1. **The data IS in the DB** — atomic facts contain "sunset", "figurines", "shoes", "two younger kids", etc.
2. **Retrieval isn't the bottleneck** — FTS5+CrossEncoder finds relevant chunks. Adding more retrieval signals (embeddings, NEAR) adds noise
3. **The problem is the answer generation step** — model can't aggregate scattered facts correctly
4. **Profile summarization loses details** — "has multiple kids" instead of "3 children"
5. **gpt-4.1-mini is the right model** — concise output matches benchmark format better than bigger models
6. **Backboard skips adversarial** — their 90.1% is on 4 categories. Our 82.4% includes adversarial
7. **Without adversarial, our score would be higher** — adversarial is ~17% of questions

---

## COMPETITION ANALYSIS

### Backboard (90.1% overall, categories 1-4 only)
- Single-hop: 89.4%, Multi-hop: 75.0%, Open-domain: 91.2%, Temporal: 91.9%
- Uses Gemini 2.5 Pro, skips adversarial, commercial API
- Turn-by-turn incremental memory formation

### Hindsight (89.6% overall, academic)
- 4 memory networks: world facts, experiences, entity summaries, beliefs
- 4-way parallel retrieval: semantic + BM25 + graph + temporal → RRF → reranker
- Narrative facts with temporal ranges and entity resolution
- Uses GPT-4o class model

### EverMemOS (92.3%)
- Atomic fact extraction (MemCells)
- BM25 + embedding + RRF + reranker
- Skips adversarial

---

### ⚠️ MIXED: Chunks-First for ALL Categories (v16)
- Used reranked conversation chunks as primary source for ALL categories (not just single-hop)
- Same DBs as v10.2, same models (gpt-4.1-mini answer, gpt-4.1-mini judge)
- Result: 81.4% overall (FLAT vs v10.4's 81.4%)
- Single-hop: 75.0% (was 71.9%) ✅ +3.1
- Multi-hop: 80.0% (was 78.6%) ✅ +1.4  
- Temporal: 89.2% (was 94.6%) ❌ -5.4 — BIG REGRESSION
- Adversarial: 83.0% (flat)
- Open-domain: 76.9% (flat)
- **WHY TEMPORAL REGRESSED:** Temporal questions need chronological episode timeline + temporal events table. Raw chunks lose time ordering. v10.4's temporal strategy (episodes + timeline) is better for temporal.
- **KEY LESSON:** Category-specific strategies matter. Chunks help precision (single-hop, multi-hop) but hurt time-dependent queries (temporal). Must use hybrid routing.

---

## WHAT TO TRY NEXT (not yet tested)

### Approach A: Better fact extraction during ingestion
- Structured JSON facts with entity links
- Entity-specific fact retrieval at query time (get ALL facts about Melanie, not just keyword-matched)
- Status: v15 subagent building this now (2026-03-29 ~5:30 PM)

### Approach B: Multi-pass answering for aggregation questions
- First pass: extract all candidate items from context
- Second pass: filter to only items that directly answer the question
- Would fix over-listing pattern

### Approach C: Skip adversarial for fair comparison
- Re-score v10.2 on categories 1-4 only
- Compare apples-to-apples with Backboard/Hindsight

### Approach D: Stronger model ONLY for single-hop answering
- Use gpt-4.1 (full) for single-hop only (already done in v10.2!)
- Try claude-3.5-sonnet or gpt-4o — but with STRICT conciseness prompting
- Key: test with gpt-4.1-mini as judge to maintain consistency

### Approach E: Separate counting/aggregation pipeline
- Detect "how many" / "what items" questions
- Use dedicated counting logic instead of free-form LLM answer

### Approach F: Hybrid v17 — best of v10.4 + v16
- Single-hop: v16 chunks-first (75% vs 71.9%)
- Multi-hop: v16 chunks-first (80% vs 78.6%)
- Temporal: v10.4 episodes+timeline (94.6% vs v16's 89.2%)
- Adversarial: v10.4 masked (83% both — keep as-is)
- Open-domain: v10.4 inference (77% both — keep as-is)
- Expected: ~84-85% overall (captures both gains without temporal regression)
- **STATUS: COMPLETE — 82.9% overall, NEW BEST (+1.5% over v10.4)**
- Single-hop: 71.9% (flat — did NOT get v16's +3.1% boost, unclear why)
- Temporal: 91.9% (recovered from v16's 89.2%, close to v10.4's 94.6%)
- Multi-hop: 84.3% (was 78.6%) ✅ +5.7 — BIGGEST WIN
- Adversarial: 83.0% (flat)
- Open-domain: 76.9% (flat)
- **KEY LESSON:** Chunks-first helps multi-hop massively. Temporal must use episodes+timeline. Single-hop gains from v16 were likely noise (didn't replicate in v17 despite same code).
- **MYSTERY:** v16 single-hop 75% vs v17 single-hop 72% — same chunks-first code. LLM nondeterminism at temperature=0? Or v16 conv-26 early results were lucky.

### ⚠️ MIXED: Two-Pass Answering + Entity Atomic Facts (v18) — 81.9%
- **Research:** Studied Hindsight (89.6%) architecture — key: narrative fact preservation, entity resolution, 4-way retrieval
- **Insight:** Our single-hop failures are 5/9 over-listing + 4/9 too vague. These are ANSWERING problems, not retrieval.
- **v18 approach:**
  1. Pass 1: Generate answer from chunks + entity-specific atomic facts
  2. Pass 2: Verification — for list questions, filter answer to ONLY items matching question scope
  3. Entity-specific atomic facts: retrieve ALL facts about target entity (not just keyword-matched)
  4. Stronger prompting: "Never paraphrase specific nouns", explicit counting rules
- **Base:** v17 (keeps temporal=episodes, multi-hop=chunks-first)
- **Result: 81.9% overall — REGRESSION from v17's 82.9%**
  - Single-hop: 75.0% (was 71.9%) ✅ +3.1
  - Temporal: 94.6% (was 91.9%) ✅ +2.7
  - Multi-hop: 80.0% (was 84.3%) ❌ -4.3
  - Open-domain: 76.9% (was 83.0%) ❌ -6.1 (only 13 questions — high variance)
- **WHY MULTI-HOP REGRESSED:** Two-pass verification + entity-specific facts make multi-hop answers TOO detailed. Model adds extra context that gets penalized. Questions like "What did they paint on Oct 13?" get confused by too many entity facts. Sub-question decomposition over-complicates when chunks already have the answer.
- **KEY LESSON:** Two-pass helps simple single-hop (precision filtering works), hurts complex multi-hop (adds noise). Verification step may be stripping relevant nuance from multi-hop answers.
- **STATUS: COMPLETE — v17 (82.9%) remains best**
