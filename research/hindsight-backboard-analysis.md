# Hindsight & Backboard Architecture Analysis

## Goal: Understand how top systems achieve 89-90% on LoCoMo, especially single-hop

---

## Backboard.io (90.1% overall, commercial)

### Per-Category Scores:
| Category | Score |
|---|---|
| Single-hop | **89.36%** |
| Multi-hop | 75.00% |
| Open-domain | **91.20%** |
| Temporal | **91.90%** |
| **Overall** | **90.00%** |

### Key Architecture Details:
- **LLM:** Gemini 2.5 Pro (for memory formation + answering)
- **Judge:** GPT-4.1
- **Per-conversation isolation:** Creates a separate assistant for each conversation
- **Skips adversarial (category 5)** — only evaluates categories 1-4
- **Memory formation:** Ingests turns one-by-one, builds internal memory as it goes
- **Multi-strategy retrieval:** Uses `memory="auto"` in their API
- **Commercial API** — architecture not published, but benchmark code is open

### Critical Insight:
Backboard gets **89.36% single-hop** — 17 points above our best (71.9%). Their multi-hop is actually WORSE than ours (75% vs 82.9%). So they optimized heavily for fact retrieval precision over multi-hop reasoning.

### What They Do That We Don't:
1. **Gemini 2.5 Pro** for answering (much stronger than gpt-4.1-mini)
2. **Turn-by-turn ingestion** with memory formation after each turn (incremental, not batch)
3. **Skip adversarial entirely** — inflates overall score
4. **Architecture-agnostic memory** that "moves between models and sessions"

---

## Hindsight (89.61% overall, academic — Vectorize.io + Virginia Tech)

### Architecture: TEMPR + CARA

**TEMPR** (Temporal Entity Memory Priming Retrieval) — handles retain + recall:
1. **4 Memory Networks:**
   - World: objective facts ("Alice works at Google")
   - Experience: agent's own experiences ("I recommended Yosemite to Alice")
   - Opinion: subjective beliefs with confidence scores
   - Observation: entity summaries synthesized from facts
   
2. **Retain Pipeline:**
   - Extract narrative facts with temporal ranges
   - Generate embeddings
   - Entity resolution (canonical names)
   - Build 4 types of graph links: temporal, semantic, entity, causal
   - Classify each fact into the correct network

3. **Recall Pipeline (CRITICAL — this is the retrieval):**
   - **4-way parallel retrieval:** semantic search + BM25 + graph traversal + temporal filtering
   - **Reciprocal Rank Fusion (RRF)** to merge results
   - **Cross-encoder reranking** for final ordering
   - Token budget-constrained output

**CARA** (Coherent Adaptive Reasoning Agents) — handles reflect:
- Behavioral profile with disposition parameters (skepticism, literalism, empathy)
- Preference-conditioned generation
- Opinion formation and reinforcement

### Key Results:
- 89.61% with large backbone (likely GPT-4o class)
- 85.67% with open-source 20B model
- Beats full-context GPT-4o

### What Hindsight Does That We Don't:
1. **4 separate memory networks** — we mix everything together
2. **Entity resolution** — canonical name mapping (e.g., "Mel" → "Melanie")
3. **Temporal ranges on facts** — each fact has valid_from/valid_to
4. **Graph traversal retrieval** — follow entity links to find related facts
5. **4-way parallel retrieval + RRF** — we only do FTS5 + reranker
6. **Narrative fact extraction** vs our episode summaries + raw chunks

---

## Comparison: What's Killing Our Single-Hop

| Feature | MemChip v10.2 | Backboard | Hindsight |
|---|---|---|---|
| Single-hop | 71.9% | 89.4% | ~87%* |
| Retrieval | FTS5 + CrossEncoder | Unknown (API) | Semantic + BM25 + Graph + Temporal + RRF + Reranker |
| Fact storage | Profiles + episodes + raw chunks + atomic facts | Unknown | 4 networks (world/experience/opinion/observation) |
| Entity resolution | None (just name matching) | Unknown | Full canonical resolution |
| Embedding search | None | Yes | Yes (semantic vector search) |
| Graph traversal | None | Unknown | Yes (entity + temporal + causal links) |

*Hindsight doesn't publish per-category LoCoMo breakdown

---

## Action Plan: What to Implement

### Phase 1: Embedding Search (biggest missing piece)
- Add sentence-transformers embedding for ALL stored facts
- Hybrid retrieval: FTS5 (BM25) + embedding similarity + RRF fusion
- This is what EVERY top system does that we don't
- Cost: zero at inference (local model), one-time embedding at ingestion

### Phase 2: Entity Resolution
- Map all name variants to canonical forms
- "Mel", "Melanie", "Melanie Smith" → "Melanie"
- This helps single-hop retrieve ALL facts about a person

### Phase 3: Narrative Facts with Temporal Ranges
- Replace episode summaries with atomic narrative facts
- Each fact gets: subject, predicate, object, valid_from, valid_to
- "Melanie bought purple running shoes" → {subject: Melanie, action: bought, object: purple running shoes, date: 2023-07-12}
- Enables temporal-aware retrieval

### Phase 4: Graph Links
- Build entity→fact and fact→fact links
- Graph traversal at recall time to find connected facts
- This is how Hindsight gets multi-hop for free

### Expected Impact:
- Phase 1 alone should push single-hop from 72% → 82-85%
- Phases 1-3 together should get us to 87-90% overall
- Phase 4 is the polish to push past 90%

---

## Critical Observations

1. **Backboard skips adversarial** — our 82.4% includes adversarial, theirs doesn't. Apples-to-oranges.
2. **Hindsight uses GPT-4o class models** — we use gpt-4.1-mini. Model quality matters.
3. **Embedding search is the #1 missing piece** — every system above 85% uses it.
4. **FTS5/BM25 alone can't do semantic matching** — "What hobbies does Melanie enjoy?" won't FTS-match "Melanie goes swimming on weekends".
